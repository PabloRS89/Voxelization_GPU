	/*
 *		r3d.c		
 *		See r3d.h for usage.
 *		Devon Powell
 *		31 August 2015
 *		This program was prepared by Los Alamos National Security, LLC at Los Alamos National
 *		Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy (DOE). 
 *		All rights in the program are reserved by the DOE and Los Alamos National Security, LLC.  
 *		Permission is granted to the public to copy and use this software without charge, provided that 
 *		this Notice and any statement of authorship are reproduced on all copies.  Neither the U.S. 
 *		Government nor LANS makes any warranty, express or implied, or assumes any liability 
 *		or responsibility for the use of this software.
 */
#include "r3d.h"
 #include "cur3d.h"
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cufft.h>

// size of the grid
#define NGRID 23
// order of polynomial integration for all tests 
#define POLY_ORDER 2

// numerical tolerances for pass/warn/fail tests
#define TOL_WARN 1.0e-8
#define TOL_FAIL 1.0e-4
#define MAX_THREADS_BLOCK 512

// minimum volume allowed for test polyhedra 
#define MIN_VOL 1.0e-8

 // forward declarations
__global__ void cur3d_vox_kernel(cur3d_element* elems, r3d_int nelem, r3d_real* rho, r3d_dvec3 n, r3d_rvec3 d);
__host__ void cur3d_err(cudaError_t err);
__device__ r3d_real cur3d_clip_and_reduce(cur3d_element tet, r3d_dvec3 gidx, r3d_rvec3 d);
__device__ void cur3du_cumsum(r3d_int* arr);
__device__ void cur3du_get_aabb(cur3d_element tet, r3d_dvec3 n, r3d_rvec3 d, r3d_dvec3 &vmin, r3d_dvec3 &vmax);
__device__ r3d_int cur3du_num_clip(cur3d_element tet, r3d_dvec3 gidx, r3d_rvec3 d);
__device__ void r3d_init_box(r3d_poly* poly, r3d_rvec3 rbounds[2]);
__device__ r3d_real cur3du_orient(cur3d_element tet);
__device__ void r3d_tet_faces_from_verts(r3d_plane* faces, r3d_rvec3* verts);

// useful macros
#define ONE_THIRD 0.333333333333333333333333333333333333333333333333333333
#define ONE_SIXTH 0.16666666666666666666666666666666666666666666666666666667
 #define CLIP_MASK 0x80
#define dot(va, vb) (va.x*vb.x + va.y*vb.y + va.z*vb.z)
#define wav(va, wa, vb, wb, vr) {			\
	vr.x = (wa*va.x + wb*vb.x)/(wa + wb);	\
	vr.y = (wa*va.y + wb*vb.y)/(wa + wb);	\
	vr.z = (wa*va.z + wb*vb.z)/(wa + wb);	\
}

#define norm(v) {					\
	r3d_real tmplen = sqrt(dot(v, v));	\
	v.x /= (tmplen + 1.0e-299);		\
	v.y /= (tmplen + 1.0e-299);		\
	v.z /= (tmplen + 1.0e-299);		\
}

clock_t t_ini,t_fin;
float milliseconds = 0;
cudaEvent_t start, stop;	    		
cublasHandle_t handle;

__host__ void cur3d_err(cudaError_t err) {
	if (err != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(err));
		//exit(0);
	}
}

// for re-indexing row-major voxel corners
__constant__ r3d_int cur3d_vv[8] = {0, 4, 3, 7, 1, 5, 2, 6};

__device__ r3d_real cur3d_reduce(r3d_poly* poly) {

	// var declarations
	r3d_real locvol;
	unsigned char v, np;
	unsigned char vcur, vnext, pnext, vstart;
	r3d_rvec3 v0, v1, v2; 

	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly->verts; 
	r3d_int* nverts = &poly->nverts; 
	
	// for keeping track of which edges have been traversed
	unsigned char emarks[R3D_MAX_VERTS][3];
	memset((void*) &emarks, 0, sizeof(emarks));

	// stack for edges
	r3d_int nvstack;
	unsigned char vstack[2*R3D_MAX_VERTS];

	// find the first unclipped vertex
	vcur = R3D_MAX_VERTS;
	for(v = 0; vcur == R3D_MAX_VERTS && v < *nverts; ++v) 
		if(!(vertbuffer[v].orient.fflags & CLIP_MASK)) vcur = v;
	
	// return if all vertices have been clipped
	if(vcur == R3D_MAX_VERTS) return 0.0;

	locvol = 0;

	// stack implementation
	nvstack = 0;
	vstack[nvstack++] = vcur;
	vstack[nvstack++] = 0;

	while(nvstack > 0) {
		
		// get the next unmarked edge
		do {
			pnext = vstack[--nvstack];
			vcur = vstack[--nvstack];
		} while(emarks[vcur][pnext] && nvstack > 0);
		if(emarks[vcur][pnext] && nvstack == 0) break; 

		// initialize face looping
		emarks[vcur][pnext] = 1;
		vstart = vcur;
		v0 = vertbuffer[vstart].pos;
		vnext = vertbuffer[vcur].pnbrs[pnext];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = (pnext+1)%3;

		// move to the second edge
		for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
		vcur = vnext;
		pnext = (np+1)%3;
		emarks[vcur][pnext] = 1;
		vnext = vertbuffer[vcur].pnbrs[pnext];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = (pnext+1)%3;

		// make a triangle fan using edges
		// and first vertex
		while(vnext != vstart) {

			v2 = vertbuffer[vcur].pos;
			v1 = vertbuffer[vnext].pos;

			locvol += ONE_SIXTH*(-(v2.x*v1.y*v0.z) + v1.x*v2.y*v0.z + v2.x*v0.y*v1.z
				   	- v0.x*v2.y*v1.z - v1.x*v0.y*v2.z + v0.x*v1.y*v2.z); 

			// move to the next edge
			for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
			vcur = vnext;
			pnext = (np+1)%3;
			emarks[vcur][pnext] = 1;
			vnext = vertbuffer[vcur].pnbrs[pnext];
			vstack[nvstack++] = vcur;
			vstack[nvstack++] = (pnext+1)%3;
		}
	}
	return locvol;
}


__device__ r3d_real cur3d_clip_and_reduce(cur3d_element tet, r3d_dvec3 gidx, r3d_rvec3 d) {

	//r3d_real moments[10];
	r3d_poly poly;
	r3d_plane faces[4];
	r3d_real gor;
	r3d_rvec3 rbounds[2] = {
		{-0.5*d.x, -0.5*d.y, -0.5*d.z}, 
		{0.5*d.x, 0.5*d.y, 0.5*d.z} 
	};
	r3d_int v, f, ii, jj, kk;
	r3d_real tetvol;
	r3d_rvec3 gpt;
	unsigned char andcmp;

	tetvol = cur3du_orient(tet);
	if(tetvol < 0.0) {
		gpt = tet.pos[2];
		tet.pos[2] = tet.pos[3];
		tet.pos[3] = gpt;
		tetvol = -tetvol;
	}
	r3d_tet_faces_from_verts(faces, tet.pos);

	// test the voxel against tet faces
	for(ii = 0; ii < 2; ++ii)
	for(jj = 0; jj < 2; ++jj)
	for(kk = 0; kk < 2; ++kk) {
		gpt.x = (ii + gidx.i)*d.x; gpt.y = (jj + gidx.j)*d.y; gpt.z = (kk + gidx.k)*d.z;
		v = cur3d_vv[4*ii + 2*jj + kk];
		poly.verts[v].orient.fflags = 0x00;
		for(f = 0; f < 4; ++f) {
			gor = faces[f].d + dot(gpt, faces[f].n);
			if(gor > 0.0) poly.verts[v].orient.fflags |= (1 << f);
			poly.verts[v].orient.fdist[f] = gor;
		}
	}

	andcmp = 0x0f;
	for(v = 0; v < 8; ++v) 
		andcmp &= poly.verts[v].orient.fflags;

	r3d_init_box(&poly, rbounds);

	//// CLIP /////

	// variable declarations
	r3d_int nvstack;
	unsigned char vstack[4*R3D_MAX_VERTS];
	unsigned char ff, np, vcur, vprev, firstnewvert, prevnewvert;
	unsigned char fmask, ffmask;

	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly.verts; 
	r3d_int* nverts = &poly.nverts; 
			
	for(f = 0; f < 4; ++f) {

		// go to the next active clip face
		fmask = (1 << f);
		while((andcmp & fmask) && f < 4)
			fmask = (1 << ++f);
		if(f == 4) break;

		// find the first vertex lying outside of the face
		// only need to find one (taking advantage of convexity)
		vcur = R3D_MAX_VERTS;
		for(v = 0; vcur == R3D_MAX_VERTS && v < *nverts; ++v) 
			if(!(vertbuffer[v].orient.fflags & (CLIP_MASK | fmask))) vcur = v;
		if(vcur == R3D_MAX_VERTS) continue; // TODO: can we do better here in terms of warp divergence?
		
		// push the first three edges and mark the starting vertex
		// as having been clipped
		nvstack = 0;
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = vertbuffer[vcur].pnbrs[1];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = vertbuffer[vcur].pnbrs[0];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = vertbuffer[vcur].pnbrs[2];
		vertbuffer[vcur].orient.fflags |= CLIP_MASK;
		firstnewvert = *nverts;
		prevnewvert = R3D_MAX_VERTS; 

		// traverse edges and clip
		// this is ordered very carefully to preserve edge connectivity
		while(nvstack > 0) {

			// get the next unclipped vertex
			do {
				vcur = vstack[--nvstack];
				vprev = vstack[--nvstack];
			} while((vertbuffer[vcur].orient.fflags & CLIP_MASK) && nvstack > 0);
			if((vertbuffer[vcur].orient.fflags & CLIP_MASK) && nvstack == 0) break; 

			// check whether this vertex is inside the face
			// if so, clip the edge and push the new vertex to vertbuffer
			if(vertbuffer[vcur].orient.fflags & fmask) {

				// compute the intersection point using a weighted
				// average of perpendicular distances to the plane
				wav(vertbuffer[vcur].pos, -vertbuffer[vprev].orient.fdist[f],
					vertbuffer[vprev].pos, vertbuffer[vcur].orient.fdist[f],
					vertbuffer[*nverts].pos);

				// doubly link to vcur
				for(np = 0; np < 3; ++np) if(vertbuffer[vcur].pnbrs[np] == vprev) break;
				vertbuffer[vcur].pnbrs[np] = *nverts;
				vertbuffer[*nverts].pnbrs[0] = vcur;

				// doubly link to previous new vert
				vertbuffer[*nverts].pnbrs[2] = prevnewvert; 
				vertbuffer[prevnewvert].pnbrs[1] = *nverts;

				// do face intersections and flags
				vertbuffer[*nverts].orient.fflags = 0x00;
				for(ff = f + 1; ff < 4; ++ff) {

					// TODO: might not need this one...
					/*ffmask = (1 << ff);*/
					/*while((andcmp & ffmask) && ff < 4)*/
						/*ffmask = (1 << ++ff);*/
					/*if(ff == 4) break;*/

					// skip if all verts are inside ff
					ffmask = (1 << ff); 
					if(andcmp & ffmask) continue;

					// weighted average keeps us in a relative coordinate system
					vertbuffer[*nverts].orient.fdist[ff] = 
							(vertbuffer[vprev].orient.fdist[ff]*vertbuffer[vcur].orient.fdist[f] 
							- vertbuffer[vprev].orient.fdist[f]*vertbuffer[vcur].orient.fdist[ff])
							/(vertbuffer[vcur].orient.fdist[f] - vertbuffer[vprev].orient.fdist[f]);
					if(vertbuffer[*nverts].orient.fdist[ff] > 0.0) vertbuffer[*nverts].orient.fflags |= ffmask;
				}

				prevnewvert = (*nverts)++;
			}
			else {

				// otherwise, determine the left and right vertices
				// (ordering is important) and push to the traversal stack
				for(np = 0; np < 3; ++np) if(vertbuffer[vcur].pnbrs[np] == vprev) break;

				// mark the vertex as having been clipped
				vertbuffer[vcur].orient.fflags |= CLIP_MASK;

				// push the next verts to the stack
				vstack[nvstack++] = vcur;
				vstack[nvstack++] = vertbuffer[vcur].pnbrs[(np+2)%3];
				vstack[nvstack++] = vcur;
				vstack[nvstack++] = vertbuffer[vcur].pnbrs[(np+1)%3];
			}
		}

		// close the clipped face
		vertbuffer[firstnewvert].pnbrs[2] = *nverts-1;
		vertbuffer[prevnewvert].pnbrs[1] = firstnewvert;
	}

	////// REDUCE ///////

#if 0
	// var declarations
	r3d_real locvol;
	unsigned char m;
	unsigned char vnext, pnext, vstart;
	r3d_rvec3 v0, v1, v2; 

	r3d_int polyorder = 0;
	
	// for keeping track of which edges have been traversed
	unsigned char emarks[R3D_MAX_VERTS][3];
	memset((void*) &emarks, 0, sizeof(emarks));

	// zero the moments
	for(m = 0; m < 10; ++m)
		moments[m] = 0.0;

	// find the first unclipped vertex
	vcur = R3D_MAX_VERTS;
	for(v = 0; vcur == R3D_MAX_VERTS && v < *nverts; ++v) 
		if(!(vertbuffer[v].orient.fflags & CLIP_MASK)) vcur = v;
	
	// return if all vertices have been clipped
	if(vcur == R3D_MAX_VERTS) return 0.0;

	// stack implementation
	nvstack = 0;
	vstack[nvstack++] = vcur;
	vstack[nvstack++] = 0;

	while(nvstack > 0) {
		
		pnext = vstack[--nvstack];
		vcur = vstack[--nvstack];

		// skip this edge if we have marked it
		if(emarks[vcur][pnext]) continue;


		// initialize face looping
		emarks[vcur][pnext] = 1;
		vstart = vcur;
		v0 = vertbuffer[vstart].pos;
		vnext = vertbuffer[vcur].pnbrs[pnext];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = (pnext+1)%3;

		// move to the second edge
		for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
		vcur = vnext;
		pnext = (np+1)%3;
		emarks[vcur][pnext] = 1;
		vnext = vertbuffer[vcur].pnbrs[pnext];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = (pnext+1)%3;

		// make a triangle fan using edges
		// and first vertex
		while(vnext != vstart) {

			v2 = vertbuffer[vcur].pos;
			v1 = vertbuffer[vnext].pos;

			locvol = ONE_SIXTH*(-(v2.x*v1.y*v0.z) + v1.x*v2.y*v0.z + v2.x*v0.y*v1.z
				   	- v0.x*v2.y*v1.z - v1.x*v0.y*v2.z + v0.x*v1.y*v2.z); 

			moments[0] += locvol; 
			if(polyorder >= 1) {
				moments[1] += locvol*0.25*(v0.x + v1.x + v2.x);
				moments[2] += locvol*0.25*(v0.y + v1.y + v2.y);
				moments[3] += locvol*0.25*(v0.z + v1.z + v2.z);
			}
			if(polyorder >= 2) {
				moments[4] += locvol*0.1*(v0.x*v0.x + v1.x*v1.x + v2.x*v2.x + v1.x*v2.x + v0.x*(v1.x + v2.x));
				moments[5] += locvol*0.1*(v0.y*v0.y + v1.y*v1.y + v2.y*v2.y + v1.y*v2.y + v0.y*(v1.y + v2.y));
				moments[6] += locvol*0.1*(v0.z*v0.z + v1.z*v1.z + v2.z*v2.z + v1.z*v2.z + v0.z*(v1.z + v2.z));
				moments[7] += locvol*0.05*(v2.x*v0.y + v2.x*v1.y + 2*v2.x*v2.y + v0.x*(2*v0.y + v1.y + v2.y) + v1.x*(v0.y + 2*v1.y + v2.y));
				moments[8] += locvol*0.05*(v2.y*v0.z + v2.y*v1.z + 2*v2.y*v2.z + v0.y*(2*v0.z + v1.z + v2.z) + v1.y*(v0.z + 2*v1.z + v2.z));
				moments[9] += locvol*0.05*(v2.x*v0.z + v2.x*v1.z + 2*v2.x*v2.z + v0.x*(2*v0.z + v1.z + v2.z) + v1.x*(v0.z + 2*v1.z + v2.z));
			}

			// move to the next edge
			for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
			vcur = vnext;
			pnext = (np+1)%3;
			emarks[vcur][pnext] = 1;
			vnext = vertbuffer[vcur].pnbrs[pnext];
			vstack[nvstack++] = vcur;
			vstack[nvstack++] = (pnext+1)%3;
		}
	}
#endif

	return tet.mass/(tetvol + 1.0e-99)*cur3d_reduce(&poly)/(d.x*d.y*d.z);
}

// parallel prefix scan in shared memory
// scan is in-place, so the result replaces the input array
// assumes input of length THREADS_PER_SM
// from GPU Gems 3, ch. 39
__device__ void cur3du_cumsum(r3d_int* arr) {

	// TODO: faster scan operation might be needed
	// (i.e. naive but less memory-efficient)
	r3d_int offset, d, ai, bi, t;

	// build the sum in place up the tree
	offset = 1;
	for (d = THREADS_PER_SM>>1; d > 0; d >>= 1) {
		__syncthreads();
		if (threadIdx.x < d) {
			ai = offset*(2*threadIdx.x+1)-1;
			bi = offset*(2*threadIdx.x+2)-1;
			arr[bi] += arr[ai];
		}
		offset *= 2;
	}

	// clear the last element
	if (threadIdx.x == 0)
		arr[THREADS_PER_SM - 1] = 0;   

	// traverse down the tree building the scan in place
	for (d = 1; d < THREADS_PER_SM; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (threadIdx.x < d) {
			ai = offset*(2*threadIdx.x+1)-1;
			bi = offset*(2*threadIdx.x+2)-1;
			t = arr[ai];
			arr[ai] = arr[bi];
			arr[bi] += t;
		}
	}
	__syncthreads();
}

__device__ void cur3du_get_aabb(cur3d_element tet, r3d_dvec3 n, r3d_rvec3 d, r3d_dvec3 &vmin, r3d_dvec3 &vmax) {

		// get the AABB for this tet
		// and clamp to destination grid dims
		r3d_int v;
		r3d_rvec3 rmin, rmax;
		rmin.x = 1.0e10; rmin.y = 1.0e10; rmin.z = 1.0e10;
		rmax.x = -1.0e10; rmax.y = -1.0e10; rmax.z = -1.0e10;
		for(v = 0; v < 4; ++v) {
			if(tet.pos[v].x < rmin.x) rmin.x = tet.pos[v].x;
			if(tet.pos[v].x > rmax.x) rmax.x = tet.pos[v].x;
			if(tet.pos[v].y < rmin.y) rmin.y = tet.pos[v].y;
			if(tet.pos[v].y > rmax.y) rmax.y = tet.pos[v].y;
			if(tet.pos[v].z < rmin.z) rmin.z = tet.pos[v].z;
			if(tet.pos[v].z > rmax.z) rmax.z = tet.pos[v].z;
		}
		vmin.i = floor(rmin.x/d.x);
		vmin.j = floor(rmin.y/d.y);
		vmin.k = floor(rmin.z/d.z);
		vmax.i = ceil(rmax.x/d.x);
		vmax.j = ceil(rmax.y/d.y);
		vmax.k = ceil(rmax.z/d.z);
		if(vmin.i < 0) vmin.i = 0;
		if(vmin.j < 0) vmin.j = 0;
		if(vmin.k < 0) vmin.k = 0;
		if(vmax.i > n.i) vmax.i = n.i;
		if(vmax.j > n.j) vmax.j = n.j;
		if(vmax.k > n.k) vmax.k = n.k;

}

__device__ r3d_int cur3du_num_clip(cur3d_element tet, r3d_dvec3 gidx, r3d_rvec3 d) {

	r3d_real tetvol;
	r3d_plane faces[4];
	r3d_rvec3 gpt;
	r3d_int f, ii, jj, kk;
	unsigned char andcmp, orcmp, fflags;
	/*r3d_int nclip;*/

	// properly orient the tet
	tetvol = cur3du_orient(tet);
	if(tetvol < 0.0) {
		gpt = tet.pos[2];
		tet.pos[2] = tet.pos[3];
		tet.pos[3] = gpt;
		tetvol = -tetvol;
	}

	// TODO: This does some sqrts that might not be needed...
	r3d_tet_faces_from_verts(faces, tet.pos);
	
	// test the bin corners against tet faces to determine voxel type
	orcmp = 0x00;
	andcmp = 0x0f;
	for(ii = 0; ii < 2; ++ii)
	for(jj = 0; jj < 2; ++jj)
	for(kk = 0; kk < 2; ++kk) {
		gpt.x = (ii + gidx.i)*d.x; gpt.y = (jj + gidx.j)*d.y; gpt.z = (kk + gidx.k)*d.z;
		fflags = 0x00;
		for(f = 0; f < 4; ++f) 
			if(faces[f].d + dot(gpt, faces[f].n) > 0.0) fflags |= (1 << f);
		andcmp &= fflags;
		orcmp |= fflags;
	}

	// if the voxel is completely outside the tet, return -1
	if(orcmp < 0x0f) return -1;
	
	// else, return the number of faces to be clipped against
	return 4 - __popc(andcmp);
}

 r3d_int r3d_is_good(r3d_poly* poly) {

	r3d_int v, np, rcur;
	r3d_int nvstack;
	r3d_int va, vb, vc;
	r3d_int vct[R3D_MAX_VERTS];
	r3d_int stack[R3D_MAX_VERTS];
	r3d_int regions[R3D_MAX_VERTS];

	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly->verts; 
	r3d_int* nverts = &poly->nverts; 

	// consistency check
	memset(&vct, 0, sizeof(vct));
	for(v = 0; v < *nverts; ++v) {

		// return false if two vertices are connected by more than one edge
		// or if any edges are obviously invalid
		for(np = 0; np < 3; ++np) {
			if(vertbuffer[v].pnbrs[np] == vertbuffer[v].pnbrs[(np+1)%3]) {
				printf("Double edge.\n");
				return 0;
			}
			if(vertbuffer[v].pnbrs[np] >= *nverts) {
				printf("Bad pointer.\n");
				return 0;
			}
		}
		vct[vertbuffer[v].pnbrs[0]]++;
		vct[vertbuffer[v].pnbrs[1]]++;
		vct[vertbuffer[v].pnbrs[2]]++;
	}
	
	// return false if any vertices are pointed to 
	// by more or fewer than three other vertices
	for(v = 0; v < *nverts; ++v) if(vct[v] != 3) {
		printf("Bad edge count: count[%d] = %d.\n", v, vct[v]);
		return 0;
	}

	// check for 3-vertex-connectedness
	// this is O(nverts^2)
	// handle multiply-connected polyhedra by testing each 
	// component separately. Flood-fill starting from each vertex
	// to give each connected region a unique ID.
	rcur = 1;
	memset(&regions, 0, sizeof(regions));
	for(v = 0; v < *nverts; ++v) {
		if(regions[v]) continue;
		nvstack = 0;
		stack[nvstack++] = v;
		while(nvstack > 0) {
			vc = stack[--nvstack];
			if(regions[vc]) continue;
			regions[vc] = rcur;
			stack[nvstack++] = vertbuffer[vc].pnbrs[0];
			stack[nvstack++] = vertbuffer[vc].pnbrs[1];
			stack[nvstack++] = vertbuffer[vc].pnbrs[2];
		}
		++rcur;
	}

	// loop over unique pairs of verts
	for(va = 0; va < *nverts; ++va) {
		rcur = regions[va];
		for(vb = va+1; vb < *nverts; ++vb) {
	
			// make sure va and vb are in the same connected component
			if(regions[vb] != rcur) continue;
	
			// pick vc != va && vc != vb 
			// and in the same connected component as va and vb
			for(vc = 0; vc < *nverts; ++vc)
			   if(regions[vc] == rcur && vc != va && vc != vb) break;
	
			// use vct to mark visited verts
			// mask out va and vb
			memset(&vct, 0, sizeof(vct));
			vct[va] = 1;
			vct[vb] = 1;
			
			// flood-fill from vc to make sure the graph is 
			// still connected when va and vb are masked
			nvstack = 0;
			stack[nvstack++] = vc;
			while(nvstack > 0) {
				vc = stack[--nvstack];
				if(vct[vc]) continue;
				vct[vc] = 1;
				stack[nvstack++] = vertbuffer[vc].pnbrs[0];
				stack[nvstack++] = vertbuffer[vc].pnbrs[1];
				stack[nvstack++] = vertbuffer[vc].pnbrs[2];
			}
	
			// if any verts in the region rcur were untouched, 
			// the graph is only 2-vertex-connected and hence an invalid polyhedron
			for(v = 0; v < *nverts; ++v) if(regions[v] == rcur && !vct[v]) {
				printf("Not 3-vertex-connected.\n");
				return 0;
			}
		}
	}	
	return 1;
}

void r3d_rotate(r3d_poly* poly, r3d_real theta, r3d_int axis) {
	r3d_int v;
	r3d_rvec3 tmp;
	r3d_real sine = sin(theta);
	r3d_real cosine = cos(theta);
	for(v = 0; v < poly->nverts; ++v) {
		tmp = poly->verts[v].pos;
		poly->verts[v].pos.xyz[(axis+1)%3] = cosine*tmp.xyz[(axis+1)%3] - sine*tmp.xyz[(axis+2)%3]; 
		poly->verts[v].pos.xyz[(axis+2)%3] = sine*tmp.xyz[(axis+1)%3] + cosine*tmp.xyz[(axis+2)%3]; 
	}
}

void r3d_translate(r3d_poly* poly, r3d_rvec3 shift) {
	r3d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		poly->verts[v].pos.x += shift.x;
		poly->verts[v].pos.y += shift.y;
		poly->verts[v].pos.z += shift.z;
	}
}

void r3d_scale(r3d_poly* poly, r3d_real scale) {
	r3d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		poly->verts[v].pos.x *= scale;
		poly->verts[v].pos.y *= scale;
		poly->verts[v].pos.z *= scale;
	}
}

void r3d_shear(r3d_poly* poly, r3d_real shear, r3d_int axb, r3d_int axs) {
	r3d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		poly->verts[v].pos.xyz[axb] += shear*poly->verts[v].pos.xyz[axs];
	}
}

void r3d_affine(r3d_poly* poly, r3d_real mat[4][4]) {
	r3d_int v;
	r3d_rvec3 tmp;
	r3d_real w;
	for(v = 0; v < poly->nverts; ++v) {
		tmp = poly->verts[v].pos;

		// affine transformation
		poly->verts[v].pos.x = tmp.x*mat[0][0] + tmp.y*mat[0][1] + tmp.z*mat[0][2] + mat[0][3];
		poly->verts[v].pos.y = tmp.x*mat[1][0] + tmp.y*mat[1][1] + tmp.z*mat[1][2] + mat[1][3];
		poly->verts[v].pos.z = tmp.x*mat[2][0] + tmp.y*mat[2][1] + tmp.z*mat[2][2] + mat[2][3];
		w = tmp.x*mat[3][0] + tmp.y*mat[3][1] + tmp.z*mat[3][2] + mat[3][3];
	
		// homogeneous divide if w != 1, i.e. in a perspective projection
		poly->verts[v].pos.x /= w;
		poly->verts[v].pos.y /= w;
		poly->verts[v].pos.z /= w;
	}
}

void r3d_init_tet(r3d_poly* poly, r3d_rvec3 verts[4]) {		
	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly->verts; 
	r3d_int* nverts = &poly->nverts; 
	
	// initialize graph connectivity
	*nverts = 4;
	vertbuffer[0].pnbrs[0] = 1;	
	vertbuffer[0].pnbrs[1] = 3;	
	vertbuffer[0].pnbrs[2] = 2;

	vertbuffer[1].pnbrs[0] = 2;	
	vertbuffer[1].pnbrs[1] = 3;	
	vertbuffer[1].pnbrs[2] = 0;	

	vertbuffer[2].pnbrs[0] = 0;	
	vertbuffer[2].pnbrs[1] = 3;	
	vertbuffer[2].pnbrs[2] = 1;	
	
	vertbuffer[3].pnbrs[0] = 1;	
	vertbuffer[3].pnbrs[1] = 2;	
	vertbuffer[3].pnbrs[2] = 0;	

	// copy vertex coordinates
	r3d_int v;
	for(v = 0; v < 4; ++v) vertbuffer[v].pos = verts[v];
}

__device__ void r3d_init_box(r3d_poly* poly, r3d_rvec3 rbounds[2]) {
	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly->verts; 
	r3d_int* nverts = &poly->nverts; 
	
	*nverts = 8;
	vertbuffer[0].pnbrs[0] = 1;	
	vertbuffer[0].pnbrs[1] = 4;	
	vertbuffer[0].pnbrs[2] = 3;	
	vertbuffer[1].pnbrs[0] = 2;	
	vertbuffer[1].pnbrs[1] = 5;	
	vertbuffer[1].pnbrs[2] = 0;	
	vertbuffer[2].pnbrs[0] = 3;	
	vertbuffer[2].pnbrs[1] = 6;	
	vertbuffer[2].pnbrs[2] = 1;	
	vertbuffer[3].pnbrs[0] = 0;	
	vertbuffer[3].pnbrs[1] = 7;	
	vertbuffer[3].pnbrs[2] = 2;	
	vertbuffer[4].pnbrs[0] = 7;	
	vertbuffer[4].pnbrs[1] = 0;	
	vertbuffer[4].pnbrs[2] = 5;	
	vertbuffer[5].pnbrs[0] = 4;	
	vertbuffer[5].pnbrs[1] = 1;	
	vertbuffer[5].pnbrs[2] = 6;	
	vertbuffer[6].pnbrs[0] = 5;	
	vertbuffer[6].pnbrs[1] = 2;	
	vertbuffer[6].pnbrs[2] = 7;	
	vertbuffer[7].pnbrs[0] = 6;	
	vertbuffer[7].pnbrs[1] = 3;	
	vertbuffer[7].pnbrs[2] = 4;	
	vertbuffer[0].pos.x = rbounds[0].x; 
	vertbuffer[0].pos.y = rbounds[0].y; 
	vertbuffer[0].pos.z = rbounds[0].z; 
	vertbuffer[1].pos.x = rbounds[1].x; 
	vertbuffer[1].pos.y = rbounds[0].y; 
	vertbuffer[1].pos.z = rbounds[0].z; 
	vertbuffer[2].pos.x = rbounds[1].x; 
	vertbuffer[2].pos.y = rbounds[1].y; 
	vertbuffer[2].pos.z = rbounds[0].z; 
	vertbuffer[3].pos.x = rbounds[0].x; 
	vertbuffer[3].pos.y = rbounds[1].y; 
	vertbuffer[3].pos.z = rbounds[0].z; 
	vertbuffer[4].pos.x = rbounds[0].x; 
	vertbuffer[4].pos.y = rbounds[0].y; 
	vertbuffer[4].pos.z = rbounds[1].z; 
	vertbuffer[5].pos.x = rbounds[1].x; 
	vertbuffer[5].pos.y = rbounds[0].y; 
	vertbuffer[5].pos.z = rbounds[1].z; 
	vertbuffer[6].pos.x = rbounds[1].x; 
	vertbuffer[6].pos.y = rbounds[1].y; 
	vertbuffer[6].pos.z = rbounds[1].z; 
	vertbuffer[7].pos.x = rbounds[0].x; 
	vertbuffer[7].pos.y = rbounds[1].y; 
	vertbuffer[7].pos.z = rbounds[1].z; 
}

 void r3d_init_poly(r3d_poly* poly, r3d_rvec3* vertices, r3d_int numverts, r3d_int** faceinds, r3d_int* numvertsperface, r3d_int numfaces) {
	// dummy vars
	r3d_int v, vprev, vcur, vnext, f, np;

	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly->verts; 
	r3d_int* nverts = &poly->nverts; 

	// count up the number of faces per vertex
	// and act accordingly
	r3d_int eperv[R3D_MAX_VERTS];
	r3d_int minvperf = R3D_MAX_VERTS;
	r3d_int maxvperf = 0;
	memset(&eperv, 0, sizeof(eperv));
	for(f = 0; f < numfaces; ++f)
		for(v = 0; v < numvertsperface[f]; ++v)
			++eperv[faceinds[f][v]];
	for(v = 0; v < numverts; ++v) {
		if(eperv[v] < minvperf) minvperf = eperv[v];
		if(eperv[v] > maxvperf) maxvperf = eperv[v];
	}

	// clear the poly
	*nverts = 0;

	// return if we were given an invalid poly
	if(minvperf < 3) return;

	if(maxvperf == 3) {
		// simple case with no need for duplicate vertices
		// read in vertex locations
		*nverts = numverts;
		for(v = 0; v < *nverts; ++v) {
			vertbuffer[v].pos = vertices[v];
			for(np = 0; np < 3; ++np) vertbuffer[v].pnbrs[np] = R3D_MAX_VERTS;
		}	
		// build graph connectivity by correctly orienting half-edges for each vertex 
		for(f = 0; f < numfaces; ++f) {
			for(v = 0; v < numvertsperface[f]; ++v) {
				vprev = faceinds[f][v];
				vcur = faceinds[f][(v+1)%numvertsperface[f]];
				vnext = faceinds[f][(v+2)%numvertsperface[f]];
				for(np = 0; np < 3; ++np) {
					if(vertbuffer[vcur].pnbrs[np] == vprev) {
						vertbuffer[vcur].pnbrs[(np+2)%3] = vnext;
						break;
					}
					else if(vertbuffer[vcur].pnbrs[np] == vnext) {
						vertbuffer[vcur].pnbrs[(np+1)%3] = vprev;
						break;
					}
				}
				if(np == 3) {
					vertbuffer[vcur].pnbrs[1] = vprev;
					vertbuffer[vcur].pnbrs[0] = vnext;
				}
			}
		}
	}
	else {
		// we need to create duplicate, degenerate vertices to account for more than
		// three edges per vertex. This is complicated.

		r3d_int tface = 0;
		for(v = 0; v < numverts; ++v) tface += eperv[v];

		// need more variables
		r3d_int v0, v1, v00, v11, numunclipped;

		// we need a few extra buffers to handle the necessary operations
		r3d_vertex vbtmp[3*R3D_MAX_VERTS];
		r3d_int util[3*R3D_MAX_VERTS];
		r3d_int vstart[R3D_MAX_VERTS];

		// build vertex mappings to degenerate duplicates
		// and read in vertex locations
		*nverts = 0;
		for(v = 0; v < numverts; ++v) {
			vstart[v] = *nverts;
			for(vcur = 0; vcur < eperv[v]; ++vcur) {
				vbtmp[*nverts].pos = vertices[v];
				for(np = 0; np < 3; ++np) vbtmp[*nverts].pnbrs[np] = R3D_MAX_VERTS;
				++(*nverts);
			}	
		}
		// fill in connectivity for all duplicates
		memset(&util, 0, sizeof(util));
		for(f = 0; f < numfaces; ++f) {
			for(v = 0; v < numvertsperface[f]; ++v) {
				vprev = faceinds[f][v];
				vcur = faceinds[f][(v+1)%numvertsperface[f]];
				vnext = faceinds[f][(v+2)%numvertsperface[f]];
				vcur = vstart[vcur] + util[vcur]++;
				vbtmp[vcur].pnbrs[1] = vnext;
				vbtmp[vcur].pnbrs[2] = vprev;
			}
		}
		// link degenerate duplicates, putting them in the correct order
		// use util to mark and avoid double-processing verts
		memset(&util, 0, sizeof(util));
		for(v = 0; v < numverts; ++v) {
			for(v0 = vstart[v]; v0 < vstart[v] + eperv[v]; ++v0) {
				for(v1 = vstart[v]; v1 < vstart[v] + eperv[v]; ++v1) {
					if(vbtmp[v0].pnbrs[2] == vbtmp[v1].pnbrs[1] && !util[v0]) {
						vbtmp[v0].pnbrs[2] = v1;
						vbtmp[v1].pnbrs[0] = v0;
						util[v0] = 1;
					}
				}
			}
		}
		// complete vertex pairs
		memset(&util, 0, sizeof(util));
		for(v0 = 0; v0 < numverts; ++v0)
		for(v1 = v0 + 1; v1 < numverts; ++v1) {
			for(v00 = vstart[v0]; v00 < vstart[v0] + eperv[v0]; ++v00)
			for(v11 = vstart[v1]; v11 < vstart[v1] + eperv[v1]; ++v11) {
				if(vbtmp[v00].pnbrs[1] == v1 && vbtmp[v11].pnbrs[1] == v0 && !util[v00] && !util[v11]) {
					vbtmp[v00].pnbrs[1] = v11;
					vbtmp[v11].pnbrs[1] = v00;
					util[v00] = 1;
					util[v11] = 1;
				}
			}
		}
		// remove unnecessary dummy vertices
		memset(&util, 0, sizeof(util));
		for(v = 0; v < numverts; ++v) {
			v0 = vstart[v];
			v1 = vbtmp[v0].pnbrs[0];
			v00 = vbtmp[v0].pnbrs[2];
			v11 = vbtmp[v1].pnbrs[0];
			vbtmp[v00].pnbrs[0] = vbtmp[v0].pnbrs[1];
			vbtmp[v11].pnbrs[2] = vbtmp[v1].pnbrs[1];
			for(np = 0; np < 3; ++np) if(vbtmp[vbtmp[v0].pnbrs[1]].pnbrs[np] == v0) break;
			vbtmp[vbtmp[v0].pnbrs[1]].pnbrs[np] = v00;
			for(np = 0; np < 3; ++np) if(vbtmp[vbtmp[v1].pnbrs[1]].pnbrs[np] == v1) break;
			vbtmp[vbtmp[v1].pnbrs[1]].pnbrs[np] = v11;
			util[v0] = 1;
			util[v1] = 1;
		}
		// copy to the real vertbuffer and compress
		numunclipped = 0;
		for(v = 0; v < *nverts; ++v) {
			if(!util[v]) {
				vertbuffer[numunclipped] = vbtmp[v];
				util[v] = numunclipped++;
			}
		}
		*nverts = numunclipped;
		for(v = 0; v < *nverts; ++v) 
			for(np = 0; np < 3; ++np)
				vertbuffer[v].pnbrs[np] = util[vertbuffer[v].pnbrs[np]];
	}
}
__device__ void r3d_tet_faces_from_verts(r3d_plane* faces, r3d_rvec3* verts) {
	r3d_rvec3 tmpcent;
	faces[0].n.x = ((verts[3].y - verts[1].y)*(verts[2].z - verts[1].z) - (verts[2].y - verts[1].y)*(verts[3].z - verts[1].z));
	faces[0].n.y = ((verts[2].x - verts[1].x)*(verts[3].z - verts[1].z) - (verts[3].x - verts[1].x)*(verts[2].z - verts[1].z));
	faces[0].n.z = ((verts[3].x - verts[1].x)*(verts[2].y - verts[1].y) - (verts[2].x - verts[1].x)*(verts[3].y - verts[1].y));
	norm(faces[0].n);
	tmpcent.x = ONE_THIRD*(verts[1].x + verts[2].x + verts[3].x);
	tmpcent.y = ONE_THIRD*(verts[1].y + verts[2].y + verts[3].y);
	tmpcent.z = ONE_THIRD*(verts[1].z + verts[2].z + verts[3].z);
	faces[0].d = -dot(faces[0].n, tmpcent);

	faces[1].n.x = ((verts[2].y - verts[0].y)*(verts[3].z - verts[2].z) - (verts[2].y - verts[3].y)*(verts[0].z - verts[2].z));
	faces[1].n.y = ((verts[3].x - verts[2].x)*(verts[2].z - verts[0].z) - (verts[0].x - verts[2].x)*(verts[2].z - verts[3].z));
	faces[1].n.z = ((verts[2].x - verts[0].x)*(verts[3].y - verts[2].y) - (verts[2].x - verts[3].x)*(verts[0].y - verts[2].y));
	norm(faces[1].n);
	tmpcent.x = ONE_THIRD*(verts[2].x + verts[3].x + verts[0].x);
	tmpcent.y = ONE_THIRD*(verts[2].y + verts[3].y + verts[0].y);
	tmpcent.z = ONE_THIRD*(verts[2].z + verts[3].z + verts[0].z);
	faces[1].d = -dot(faces[1].n, tmpcent);

	faces[2].n.x = ((verts[1].y - verts[3].y)*(verts[0].z - verts[3].z) - (verts[0].y - verts[3].y)*(verts[1].z - verts[3].z));
	faces[2].n.y = ((verts[0].x - verts[3].x)*(verts[1].z - verts[3].z) - (verts[1].x - verts[3].x)*(verts[0].z - verts[3].z));
	faces[2].n.z = ((verts[1].x - verts[3].x)*(verts[0].y - verts[3].y) - (verts[0].x - verts[3].x)*(verts[1].y - verts[3].y));
	norm(faces[2].n);
	tmpcent.x = ONE_THIRD*(verts[3].x + verts[0].x + verts[1].x);
	tmpcent.y = ONE_THIRD*(verts[3].y + verts[0].y + verts[1].y);
	tmpcent.z = ONE_THIRD*(verts[3].z + verts[0].z + verts[1].z);
	faces[2].d = -dot(faces[2].n, tmpcent);

	faces[3].n.x = ((verts[0].y - verts[2].y)*(verts[1].z - verts[0].z) - (verts[0].y - verts[1].y)*(verts[2].z - verts[0].z));
	faces[3].n.y = ((verts[1].x - verts[0].x)*(verts[0].z - verts[2].z) - (verts[2].x - verts[0].x)*(verts[0].z - verts[1].z));
	faces[3].n.z = ((verts[0].x - verts[2].x)*(verts[1].y - verts[0].y) - (verts[0].x - verts[1].x)*(verts[2].y - verts[0].y));
	norm(faces[3].n);
	tmpcent.x = ONE_THIRD*(verts[0].x + verts[1].x + verts[2].x);
	tmpcent.y = ONE_THIRD*(verts[0].y + verts[1].y + verts[2].y);
	tmpcent.z = ONE_THIRD*(verts[0].z + verts[1].z + verts[2].z);
	faces[3].d = -dot(faces[3].n, tmpcent);
}

 void r3d_box_faces_from_verts(r3d_plane* faces, r3d_rvec3* rbounds) {
	faces[0].n.x = 0.0; faces[0].n.y = 0.0; faces[0].n.z = 1.0; faces[0].d = rbounds[0].z; 
	faces[2].n.x = 0.0; faces[2].n.y = 1.0; faces[2].n.z = 0.0; faces[2].d = rbounds[0].y; 
	faces[4].n.x = 1.0; faces[4].n.y = 0.0; faces[4].n.z = 0.0; faces[4].d = rbounds[0].x; 
	faces[1].n.x = 0.0; faces[1].n.y = 0.0; faces[1].n.z = -1.0; faces[1].d = rbounds[1].z; 
	faces[3].n.x = 0.0; faces[3].n.y = -1.0; faces[3].n.z = 0.0; faces[3].d = rbounds[1].y; 
	faces[5].n.x = -1.0; faces[5].n.y = 0.0; faces[5].n.z = 0.0; faces[5].d = rbounds[1].x; 
}

 void r3d_poly_faces_from_verts(r3d_plane* faces, r3d_rvec3* vertices, r3d_int numverts, r3d_int** faceinds, r3d_int* numvertsperface, r3d_int numfaces) {
	// dummy vars
	r3d_int v, f;
	r3d_rvec3 p0, p1, p2, centroid;

	// calculate a centroid and a unit normal for each face 
	for(f = 0; f < numfaces; ++f) {
		centroid.x = 0.0;
		centroid.y = 0.0;
		centroid.z = 0.0;
		faces[f].n.x = 0.0;
		faces[f].n.y = 0.0;
		faces[f].n.z = 0.0;
		
		for(v = 0; v < numvertsperface[f]; ++v) {
			// add cross product of edges to the total normal
			p0 = vertices[faceinds[f][v]];
			p1 = vertices[faceinds[f][(v+1)%numvertsperface[v]]];
			p2 = vertices[faceinds[f][(v+2)%numvertsperface[v]]];
			faces[f].n.x += (p1.y - p0.y)*(p2.z - p0.z) - (p1.z - p0.z)*(p2.y - p0.y);
			faces[f].n.y += (p1.z - p0.z)*(p2.x - p0.x) - (p1.x - p0.x)*(p2.z - p0.z);
			faces[f].n.z += (p1.x - p0.x)*(p2.y - p0.y) - (p1.y - p0.y)*(p2.x - p0.x);
			// add the vertex position to the centroid
			centroid.x += p0.x;
			centroid.y += p0.y;
			centroid.z += p0.z;
		}
		// normalize the normals and set the signed distance to origin
		centroid.x /= numvertsperface[f];
		centroid.y /= numvertsperface[f];
		centroid.z /= numvertsperface[f];
		norm(faces[f].n);
		faces[f].d = -dot(faces[f].n, centroid);
	}
}

__device__ r3d_real cur3du_orient(cur3d_element tet) {
	r3d_real adx, bdx, cdx;
	r3d_real ady, bdy, cdy;
	r3d_real adz, bdz, cdz;
	adx = tet.pos[0].x - tet.pos[3].x;
	bdx = tet.pos[1].x - tet.pos[3].x;
	cdx = tet.pos[2].x - tet.pos[3].x;
	ady = tet.pos[0].y - tet.pos[3].y;
	bdy = tet.pos[1].y - tet.pos[3].y;
	cdy = tet.pos[2].y - tet.pos[3].y;
	adz = tet.pos[0].z - tet.pos[3].z;
	bdz = tet.pos[1].z - tet.pos[3].z;
	cdz = tet.pos[2].z - tet.pos[3].z;
	
	return -ONE_SIXTH*(adx * (bdy * cdz - bdz * cdy)+ bdx * (cdy * adz - cdz * ady)	+ cdx * (ady * bdz - adz * bdy));
}

void r3d_print(r3d_poly* poly) {
	r3d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		printf("  vertex %d: pos = ( %.10e , %.10e , %.10e ), nbrs = %d %d %d\n", 
			v, poly->verts[v].pos.x, poly->verts[v].pos.y, poly->verts[v].pos.z, 
			poly->verts[v].pnbrs[0], poly->verts[v].pnbrs[1], poly->verts[v].pnbrs[2]);
	}
}

#define dot3(va, vb) (va.x*vb.x + va.y*vb.y + va.z*vb.z)
#define norm3(v) {					\
	r3d_real tmplen = sqrt(dot3(v, v));	\
	v.x /= (tmplen + 1.0e-299);		\
	v.y /= (tmplen + 1.0e-299);		\
	v.z /= (tmplen + 1.0e-299);		\
}

int rand_int(int N) {
	// random integers from 0 (incl) to N (excl)
	return rand()%N;
}	

double rand_uniform() {
	// uniform random in (0, 1)
	return ((double) rand())/RAND_MAX;
}

double rand_normal() {
	// uses a Box-Muller transform to get two normally distributed numbers
	// from two uniformly distributed ones. We throw one away here.
	double u1 = rand_uniform();
	double u2 = rand_uniform();
	return sqrt(-2.0*log(u1))*cos(6.28318530718*u2);
}

r3d_rvec3 rand_uvec_3d() {
	// generates a random, isotropically distributed unit vector
	r3d_rvec3 tmp;
	tmp.x = rand_normal();
	tmp.y = rand_normal();
	tmp.z = rand_normal();
	norm3(tmp);
	return tmp;
}
/*
r3d_real rand_tet_3d(cur3d_element tet, r3d_real minvol) {
	// generates a random tetrahedron with vertices on the unit sphere,
	// guaranteeing a volume of at least MIN_VOL (to avoid degenerate cases)
	r3d_int v;
	//v = threadIdx.x;
	r3d_rvec3 swp;
	r3d_real tetvol = 0.0;
	while(tetvol < minvol) {		
		for(v = 0; v < 4; ++v) {
			verts[v] = rand_uvec_3d();				
		}
		tetvol = cur3du_orient(tet);
		if(tetvol < 0.0) {
			swp = verts[2];
			verts[2] = verts[3];
			verts[3] = swp;
			tetvol = -tetvol;
		}
	}		
	return tetvol;
}*/

// TODO: make this a generic "split" routine that just takes a plane.
__device__ void r3d_split(r3d_poly* inpoly, r3d_poly** outpolys, r3d_real coord, r3d_int ax);

__global__ void cur3d_vox_kernel(cur3d_element* elems, r3d_int nelem, r3d_real* rho, r3d_dvec3 n, r3d_rvec3 d) {
	// voxel ring buffers, separated by number of faces to clip against
	// TODO: group clip-and-reduce operations by number of clip faces
	__shared__ r3d_int clip_voxels[4][2*THREADS_PER_SM];
	__shared__ r3d_int clip_tets[4][2*THREADS_PER_SM];
	__shared__ r3d_int vbuf_start[4], vbuf_end[4]; 

	r3d_int tag_face[4];
	r3d_int nclip;

	__shared__ r3d_int cuminds[THREADS_PER_SM];

	// cumulative voxel offsets
	__shared__ r3d_int voxel_offsets[THREADS_PER_SM];
	__shared__ r3d_int voxels_per_block;

	// working vars
	cur3d_element tet;
	r3d_dvec3 vmin, vmax, vn, gidx; // voxel index range
	r3d_int vflat; // counters and such
	r3d_int tid; // local (shared memory) tet id
	r3d_int gid; // global tet id
	r3d_int vid; // local (shared memory) voxel id
	r3d_int btm;
	r3d_int top;
	r3d_int nf;
	
	// STEP 1
	// calculate offsets of each tet in the global voxel array
	// assumes that the total tet batch is <= GPU threads
	tid = threadIdx.x;
 	gid = blockIdx.x*blockDim.x + tid;
	voxel_offsets[tid] = 0;
	if(gid < nelem) {
		cur3du_get_aabb(elems[gid], n, d, vmin, vmax);
		voxel_offsets[tid] = (vmax.i - vmin.i)*(vmax.j - vmin.j)*(vmax.k - vmin.k); 
	}
	if(threadIdx.x == blockDim.x - 1)
		voxels_per_block = voxel_offsets[threadIdx.x];
	cur3du_cumsum(voxel_offsets);
	if(threadIdx.x == blockDim.x - 1)
		voxels_per_block += voxel_offsets[threadIdx.x];
	__syncthreads();

	// STEP 2
	// process all voxels in the AABBs and bin into separate buffers
	// for face and interior voxels
	// each thread gets one voxel
	if(threadIdx.x == 0)
		for(nf = 0; nf < 4; ++nf) {
			vbuf_start[nf] = 0;
			vbuf_end[nf] = 0;
		}
	__syncthreads();
	for(vid = threadIdx.x; vid < voxels_per_block; vid += blockDim.x) {

		// binary search through cumulative voxel indices
		// to get the correct tet
		btm = 0;
		top = THREADS_PER_SM; 
		tid = (btm + top)/2;
		while(vid < voxel_offsets[tid] || vid >= voxel_offsets[tid+1]) {
			if(vid < voxel_offsets[tid]) top = tid;
			else btm = tid + 1;
			tid = (btm + top)/2;
		}
	 	gid = blockIdx.x*blockDim.x + tid;
		tet = elems[gid];

		// recompute the AABB for this tet	
		// to get the grid index of this voxel
		cur3du_get_aabb(tet, n, d, vmin, vmax);
		vn.i = vmax.i - vmin.i;
		vn.j = vmax.j - vmin.j;
		vn.k = vmax.k - vmin.k;
		vflat = vid - voxel_offsets[tid]; 
		gidx.i = vflat/(vn.j*vn.k); 
		gidx.j = (vflat - vn.j*vn.k*gidx.i)/vn.k;
		gidx.k = vflat - vn.j*vn.k*gidx.i - vn.k*gidx.j;
		gidx.i += vmin.i; gidx.j += vmin.j; gidx.k += vmin.k;

		// check the voxel against the tet faces
		nclip = cur3du_num_clip(tet, gidx, d);

		for(nf = 0; nf < 4; ++nf) tag_face[nf] = 0;
		if(nclip == 0) // completely contained voxel 
			tag_face[0] = 0;//atomicAdd(&rho[n.j*n.k*gidx.i + n.k*gidx.j + gidx.k], tet.mass/(fabs(cur3du_orient(tet)) + 1.0e-99));
		else if(nclip > 0) // voxel must be clipped
			tag_face[nclip-1] = 1;

		__syncthreads();

		// STEP 3
		// accumulate face voxels to a ring buffer
		// parallel scan to get indices, then parallel write to the ring buffer
		for(nf = 0; nf < 4; ++nf) {
			cuminds[threadIdx.x] = tag_face[nf];
			cur3du_cumsum(cuminds);
			if(tag_face[nf]) {
				clip_voxels[nf][(vbuf_end[nf] + cuminds[threadIdx.x])%(2*THREADS_PER_SM)] = n.j*n.k*gidx.i + n.k*gidx.j + gidx.k;
				clip_tets[nf][(vbuf_end[nf] + cuminds[threadIdx.x])%(2*THREADS_PER_SM)] = tid;
			}
			if(threadIdx.x == blockDim.x - 1)
				vbuf_end[nf] += cuminds[threadIdx.x] + tag_face[nf];
			__syncthreads();
		}

		// STEP 4
		// parallel reduction of face voxels (1 per thread)
		for(nf = 0; nf < 4; ++nf) {
			if(vbuf_end[nf] - vbuf_start[nf] >= THREADS_PER_SM) {

				// recompute i, j, k, faces for this voxel
				vflat = clip_voxels[nf][(threadIdx.x + vbuf_start[nf])%(2*THREADS_PER_SM)]; 
				gidx.i = vflat/(n.j*n.k); 
				gidx.j = (vflat - n.j*n.k*gidx.i)/n.k;
				gidx.k = vflat - n.j*n.k*gidx.i - n.k*gidx.j;
				tet = elems[blockIdx.x*blockDim.x + clip_tets[nf][(threadIdx.x + vbuf_start[nf])%(2*THREADS_PER_SM)]];

				// clip and reduce to grid
				/*atomicAdd(&rho[vflat], cur3d_clip_and_reduce(tet, gidx, d));*/

				// shift ring buffer head
				if(threadIdx.x == 0)
					vbuf_start[nf] += THREADS_PER_SM;
			} 
			__syncthreads();
		}
	}

	// STEP 5
	// clean up any face voxels remaining in the ring buffer
	/*	for(nf = 0; nf < 4; ++nf) {
			if(threadIdx.x < vbuf_end[nf] - vbuf_start[nf]) {

				// recompute i, j, k, faces for this voxel
				vflat = clip_voxels[nf][(threadIdx.x + vbuf_start[nf])%(2*THREADS_PER_SM)]; 
				gidx.i = vflat/(n.j*n.k); 
				gidx.j = (vflat - n.j*n.k*gidx.i)/n.k;
				gidx.k = vflat - n.j*n.k*gidx.i - n.k*gidx.j;
				tet = elems[blockIdx.x*blockDim.x + clip_tets[nf][(threadIdx.x + vbuf_start[nf])%(2*THREADS_PER_SM)]];

				// clip and reduce to grid
				atomicAdd(&rho[vflat], cur3d_clip_and_reduce(tet, gidx, d));

				// shift ring buffer head
				if(threadIdx.x == 0)
					vbuf_start[nf] += THREADS_PER_SM;
			} 
			__syncthreads();
		}


	if(threadIdx.x < vbuf_end - vbuf_start) {

		// recompute i, j, k, faces for this voxel
		vflat = face_voxels[(threadIdx.x + vbuf_start)%(2*THREADS_PER_SM)]; 
		gidx.i = vflat/(n.j*n.k); 
		gidx.j = (vflat - n.j*n.k*gidx.i)/n.k;
		gidx.k = vflat - n.j*n.k*gidx.i - n.k*gidx.j;
		tet = elems[blockIdx.x*blockDim.x + face_tets[(threadIdx.x + vbuf_start)%(2*THREADS_PER_SM)]];

		// clip and reduce to grid
		atomicAdd(&rho[vflat], cur3d_clip_and_reduce(tet, gidx, d));
	}*/
}

__device__ void r3d_split(r3d_poly* inpoly, r3d_poly** outpolys, r3d_real coord, r3d_int ax) {
	// direct access to vertex buffer
	if(inpoly->nverts <= 0) return;
	r3d_int* nverts = &inpoly->nverts;
	r3d_vertex* vertbuffer = inpoly->verts; 
	r3d_int v, np, npnxt, onv, vcur, vnext, vstart, pnext, nright, cside;
	r3d_rvec3 newpos;
	r3d_int side[R3D_MAX_VERTS];
	r3d_real sdists[R3D_MAX_VERTS];

	// calculate signed distances to the clip plane
	nright = 0;
	memset(&side, 0, sizeof(side));
	for(v = 0; v < *nverts; ++v) {			
		sdists[v] = coord - vertbuffer[v].pos.xyz[ax];
		if(sdists[v] < 0.0) {
			side[v] = 1;
			nright++;
		}
	}

	// return if the poly lies entirely on one side of it 
	if(nright == 0) {
		*(outpolys[0]) = *inpoly;
		outpolys[1]->nverts = 0;
		return;
	}
	if(nright == *nverts) {
		*(outpolys[1]) = *inpoly;
		outpolys[0]->nverts = 0;
		return;
	}

	// check all edges and insert new vertices on the bisected edges 
	onv = inpoly->nverts;
	for(vcur = 0; vcur < onv; ++vcur) {
		if(side[vcur]) continue;
		for(np = 0; np < 3; ++np) {
			vnext = vertbuffer[vcur].pnbrs[np];
			if(!side[vnext]) continue;
			wav(vertbuffer[vcur].pos, -sdists[vnext], vertbuffer[vnext].pos, sdists[vcur], newpos);
			vertbuffer[*nverts].pos = newpos;
			vertbuffer[*nverts].pnbrs[0] = vcur;
			vertbuffer[vcur].pnbrs[np] = *nverts;
			(*nverts)++;
			vertbuffer[*nverts].pos = newpos;
			side[*nverts] = 1;
			vertbuffer[*nverts].pnbrs[0] = vnext;
			for(npnxt = 0; npnxt < 3; ++npnxt) 
				if(vertbuffer[vnext].pnbrs[npnxt] == vcur) break;
			vertbuffer[vnext].pnbrs[npnxt] = *nverts;
			(*nverts)++;
		}
	}

	// for each new vert, search around the faces for its new neighbors
	// and doubly-link everything
	for(vstart = onv; vstart < *nverts; ++vstart) {
		vcur = vstart;
		vnext = vertbuffer[vcur].pnbrs[0];
		do {
			for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
			vcur = vnext;
			pnext = (np+1)%3;
			vnext = vertbuffer[vcur].pnbrs[pnext];
		} while(vcur < onv);
		vertbuffer[vstart].pnbrs[2] = vcur;
		vertbuffer[vcur].pnbrs[1] = vstart;
	}

	// copy and compress vertices into their new buffers
	// reusing side[] for reindexing
	onv = *nverts;
	outpolys[0]->nverts = 0;
	outpolys[1]->nverts = 0;
	for(v = 0; v < onv; ++v) {
		cside = side[v];
		outpolys[cside]->verts[outpolys[cside]->nverts] = vertbuffer[v];
		side[v] = (outpolys[cside]->nverts)++;
	}

	for(v = 0; v < outpolys[0]->nverts; ++v) 
		for(np = 0; np < 3; ++np)
			outpolys[0]->verts[v].pnbrs[np] = side[outpolys[0]->verts[v].pnbrs[np]];
	for(v = 0; v < outpolys[1]->nverts; ++v) 
		for(np = 0; np < 3; ++np)
			outpolys[1]->verts[v].pnbrs[np] = side[outpolys[1]->verts[v].pnbrs[np]];
}

void r3d_get_ibox(r3d_poly* poly, r3d_dvec3 ibox[2], r3d_rvec3 d) {
	r3d_int i, v;
	r3d_rvec3 rbox[2];
	for(i = 0; i < 3; ++i) {
		rbox[0].xyz[i] = 1.0e30;
		rbox[1].xyz[i] = -1.0e30;
	}
	for(v = 0; v < poly->nverts; ++v) {
		for(i = 0; i < 3; ++i) {
			if(poly->verts[v].pos.xyz[i] < rbox[0].xyz[i]) rbox[0].xyz[i] = poly->verts[v].pos.xyz[i];
			if(poly->verts[v].pos.xyz[i] > rbox[1].xyz[i]) rbox[1].xyz[i] = poly->verts[v].pos.xyz[i];
		}
	}
	for(i = 0; i < 3; ++i) {
		ibox[0].ijk[i] = floor(rbox[0].xyz[i]/d.xyz[i]);
		ibox[1].ijk[i] = ceil(rbox[1].xyz[i]/d.xyz[i]);
	}
}

//This function calculates the average of the data array over the CPU in a sequential programming
float avg_CPU(size_t size, float *pos){
	float avg = 0;
	unsigned int x;
	for(x = 0; x<size; x++){
		avg = avg + pos[x];
	}
	avg = avg / size;
	return avg;
}
//This function is a kind of helper to qsot to determine if the value is greater or lower than..
int compare(const void *a, const void *b) {
    return ( *(int*)a - *(int*)b );
}
//Function which calculates the medium value of the dataset in a sequential programming using qsort algorithm
float medium_CPU(size_t size, float *pos){
	float medium, medium1, medium2;
	int x = (int)floor(size/2);
	qsort(pos, size, sizeof(float), &compare);	
	if(size%2 == 0){
		medium1 = pos[x];
		medium2 = pos[x-1];
		medium = (medium1 + medium2) / 2;
	}else{
		medium = pos[x];
	}		
	return medium;
}
//Calculate the standard deviation in sequential programming over CPU
float StDev_CPU(size_t size, float *pos){
	float medium = 0, ed = 0;
	unsigned int x;
	for(x = 0; x<size; x++){
		medium = medium + pos[x];
	}
	medium = medium / size;		
	for(x = 0; x<size; x++){
		ed = ed + pow(pos[x]-medium, 2);
	}	
	ed = ed / size;
	ed = sqrt(ed);
	return ed;
}
//Calculate the standard deviation in sequential programming over GPU
__global__ void StDev_GPU(size_t size, float *pos, float *medium, float *ED){		
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;	
	float tmp;
	if (idx >= size) {
 		return;
   	}	
    	tmp = pow(pos[idx]-*medium,2);	
	__syncthreads();
	atomicAdd(ED, tmp);
	__syncthreads();
}

void medium_GPU(size_t size, float *pos){
	//FILE *f = fopen("times_Medium.csv", "a");
	thrust::host_vector<float> h_keys(size);		
	unsigned int i;
	for(i=0; i<size; i++){
		h_keys[i] = pos[i];
	}
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 

	thrust::device_vector<float> d_values = h_keys;
	thrust::sort(d_values.begin(), d_values.end());
	thrust::host_vector<float> h_values = d_values;

	cudaEventSynchronize(stop);
	cudaEventRecord(stop, 0);	    	    	
	cudaEventSynchronize(stop);	

	bool bTestResult = thrust::is_sorted(h_values.begin(), h_values.end());
	float m1, m2, medium;
    	int x = (int)floor(size/2);
    	if(size%2 == 0){
		m1 = h_values[x];
		m2 = h_values[x-1];
		medium = (m1 + m2) / 2;
	}else{
		medium = h_values[x];
	}
	if(bTestResult)
	{
		printf("Medium in GPU:%f\n",medium);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("\tms: %f\n",milliseconds);
		//fprintf(f, "%f", milliseconds);
		//fclose(f);
	}
	else
		printf("No sorted\n");
}

__global__ void real2Complex(float *a, cufftComplex *c, size_t N){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;	

	if(idx < N){
		int index = idx * N;
		c[index].x = a[index];		
	}
}

extern "C"{
	void calc_FFT(size_t size, float *pos){
		//printf("Calculando FFT\n");
		//FILE *f = fopen("FFT.csv", "a");
		cufftComplex *h_data = (float2 *) malloc(sizeof(float2) * size);
		cufftComplex *r_data = (float2 *) malloc(sizeof(float2) * size);

		for(unsigned int i=0; i<size; i++){
			h_data[i].x = i;
			h_data[i].y = pos[i];
		}

		cufftComplex *d_data;
		cudaMalloc((void **)&d_data, size*sizeof(cufftComplex));
		cudaMemcpy(d_data, h_data, size*sizeof(cufftComplex), cudaMemcpyHostToDevice);

		cufftHandle plan;
		cufftPlan1d(&plan, size, CUFFT_C2C, 1);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		cufftExecC2C(plan, (cufftComplex *)d_data, (cufftComplex *)d_data, CUFFT_FORWARD);

		cudaEventSynchronize(stop);
		cudaEventRecord(stop, 0);	    	    	
		cudaEventSynchronize(stop);		
		cudaEventElapsedTime(&milliseconds, start, stop);

		//cufftExecC2C(plan, (cufftComplex *)d_data, (cufftComplex *)d_data, CUFFT_INVERSE);

		cudaMemcpy(r_data, d_data, size*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
		
		for(unsigned int i=0; i<10; i++){
			pos[i] = r_data[i].x / (float)size;		
			//pos[i] = r_data[i].x;
			//printf("%f, %f\n", r_data[i].x, r_data[i].y);
		}

		cufftDestroy(plan);
		free(h_data);		
		cudaFree(d_data);

		printf("CUFFT:\n");
		printf("\tms: %f\n",milliseconds);
		//printf(f, "%f\n", milliseconds);
		//fclose(f);		
	}
	
	//Calculate the average of the input data ´pos´
	void calc_avg(size_t size, float *pos){	
		float result, *d_pos, avg_gpu, average_cpu;
		//FILE *f = fopen("times_Average.csv", "a");
	
		cublasCreate(&handle);
		cudaMalloc((void **)&d_pos, size * sizeof(float));
		cudaMemcpy(d_pos, pos, size * sizeof(float), cudaMemcpyHostToDevice);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		cublasSasum(handle, size, d_pos, 1, &result);		
		avg_gpu = result/size;

		cudaEventSynchronize(stop);
		cudaEventRecord(stop, 0);	    	    	
		cudaEventSynchronize(stop);		
		cudaEventElapsedTime(&milliseconds, start, stop);

		cudaFree(d_pos);
		cublasDestroy(handle);
		
		printf("Average in GPU: %f\n", avg_gpu);
		printf("\tms: %f\n",milliseconds);

		t_ini=clock();	    
		average_cpu = avg_CPU(size, pos);	    
		t_fin=clock();
		printf("Average in CPU:%f\n", average_cpu);	    
		printf("\tms: %f\n\n",((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);		
		//fprintf(f, "%f %f\n", milliseconds, ((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fclose(f);      	   
	}
	
   	//Calculate the medium value   	
	void calc_medium(size_t size, float *pos){
		medium_GPU(size, pos);
		//FILE *f = fopen("times_Medium.csv", "a");
		t_ini=clock();
		float M = medium_CPU(size, pos);
		t_fin=clock();
		printf("Medium in CPU:%f\n",M);
		printf("\tms: %f\n\n",((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fprintf(f, " %f\n", ((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fclose(f);
	}
	//Calculate the standard deviation value
	void calc_StDev(size_t size, float *pos){		
		float result;
		float *d_pos, *medium, *m, *SD, *sd;
		int block, thread;
		//FILE *f = fopen("times_StDev.csv", "a");

		block = ceil(size/MAX_THREADS_BLOCK)+1;
	   	thread = MAX_THREADS_BLOCK;

		dim3 BLOCK(block);
		dim3 THREAD(thread);

		cublasCreate(&handle);
		cudaMalloc((void **)&d_pos, size * sizeof(float));
		cudaMemcpy(d_pos, pos, size * sizeof(float), cudaMemcpyHostToDevice);

		cublasSasum(handle, size, d_pos,1,&result);

		cudaFree(d_pos);
		cublasDestroy(handle);					

		cudaMalloc((void **)&d_pos, size * sizeof(float));
		cudaMalloc((void **)&medium, sizeof(float));
		cudaMalloc((void **)&SD, sizeof(float));
		m = (float *)malloc(sizeof(float));
		sd = (float *)malloc(sizeof(float));	    
		*m = result/size;
		cudaMemcpy(d_pos, pos, size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(medium, m, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(SD, sd, sizeof(float), cudaMemcpyHostToDevice);
			  
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);   	    
		StDev_GPU <<< BLOCK, THREAD >>> (size, d_pos, medium, SD);
		cudaMemcpy(sd, SD, sizeof(float), cudaMemcpyDeviceToHost);
		float sd_result = *sd;	    	    
		sd_result = sd_result / size;
		sd_result = sqrt(sd_result);
		cudaEventSynchronize(stop);
		cudaEventRecord(stop, 0);	    	    	
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);	    	    	    	    
		printf("Standar Deviation in GPU:%f\n",sd_result);	    
		printf("\tms: %f\n",milliseconds);	    
		cudaEventDestroy(start);
		cudaEventDestroy(stop);	  
		cudaFree(d_pos);  
		cudaFree(medium);
		free(m);
		
		t_ini=clock();
		float d = StDev_CPU(size, pos);
		t_fin=clock();
		printf("Standar Deviation in CPU:%f\n",d);
		printf("\tms: %f\n\n",((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fprintf(f, "%f %f\n", milliseconds, ((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fclose(f);
	}
	//Calculate the minimum and maximum value existing in dataset
	void calc_MaxMin(size_t size, float *pos){		
		int d_max, d_min;
		float *d_pos;    
		//FILE *f = fopen("times_maxmin.csv", "a");

		cublasCreate(&handle);
		cudaMalloc((void **)&d_pos, size * sizeof(float));
		cudaMemcpy(d_pos, pos, size * sizeof(float), cudaMemcpyHostToDevice);

		cudaEventCreate(&start);
	   	cudaEventCreate(&stop);
	   	cudaEventRecord(start, 0);
		
		cublasIsamax(handle, size, d_pos,1,&d_max);
		cublasIsamin(handle, size, d_pos,1,&d_min);

		cudaEventSynchronize(stop);
	    	cudaEventRecord(stop, 0);	    	    	
	    	cudaEventSynchronize(stop);		
		cudaEventElapsedTime(&milliseconds, start, stop);		

		cudaFree(d_pos);
		cublasDestroy(handle);
		printf("GPU:\n");
		printf(" - Minimum: %f\n", pos[d_min-1]);
		printf(" - Maximun: %f\n", pos[d_max-1]);
		printf("\tms: %f\n",milliseconds);

		float min = 0;
		float max = 0;
		
		t_ini=clock();
		qsort(pos, size, sizeof(float), &compare);
		min = pos[0];
		max = pos[size-1];
		t_fin=clock();

		printf("CPU:\n");
		printf(" - Minimum: %f\n", min);
		printf(" - Maximun: %f\n", max);
		printf("\tms: %f\n\n",((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fprintf(f, "%f %f\n", milliseconds, ((double)(t_fin-t_ini)/CLOCKS_PER_SEC)*1000);
		//fclose(f);		
	}

	__host__ void voxelization() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		setbuf(stdout, NULL);
		cudaError_t e = cudaSuccess;
		//Variables declaration and initialization
		r3d_int nelem = 10;
		r3d_real* rho_h = (r3d_real*)malloc(sizeof(r3d_real));
		*rho_h = 10;		
		cur3d_element* elems_h = (cur3d_element*)malloc(nelem*sizeof(cur3d_element));
		elems_h->mass = 20;
		elems_h->pos[0].x = 5;
		elems_h->pos[0].y = 5;
		elems_h->pos[0].z = 5;
		elems_h->pos[0].xyz[0] = 10;
		elems_h->pos[0].xyz[1] = 10;
		elems_h->pos[0].xyz[2] = 10;				
		r3d_dvec3 n;
		n.ijk[0] = 5;
		n.ijk[1] = 5;
		n.ijk[2] = 5;		
		r3d_rvec3 d;
		d.xyz[0] = 6;
		d.xyz[1] = 6;
		d.xyz[2] = 6;
		// Allocate target grid on the device
		//r3d_long ntot = n.i*n.j*n.k;
		r3d_long ntot = 20;
		r3d_real* rho_d;
		cudaMalloc((void**) &rho_d, ntot*sizeof(r3d_real));
		cudaMemcpy(rho_d, rho_h, ntot*sizeof(r3d_real), cudaMemcpyHostToDevice);

		// Allocate and copy element buffer to the device
		cur3d_element* elems_d;
		cudaMalloc((void**) &elems_d, nelem*sizeof(cur3d_element));
		cudaMemcpy(elems_d, elems_h, nelem*sizeof(cur3d_element), cudaMemcpyHostToDevice);

		printf("Launching voxelization kernel, %d SMs * %d threads/SM = %d threads\n", NUM_SM, THREADS_PER_SM, NUM_SM*THREADS_PER_SM);		
		
		cur3d_vox_kernel<<<NUM_SM, THREADS_PER_SM>>>(elems_d, nelem, rho_d, n, d);
		cudaEventRecord(stop);
		e = cudaGetLastError();
		cur3d_err(e);		

		// TODO: this needs to be a reduction...
		cudaMemcpy(rho_h, rho_d, ntot*sizeof(r3d_real), cudaMemcpyDeviceToHost);
		cudaMemcpy(elems_h, elems_d, nelem*sizeof(cur3d_element), cudaMemcpyDeviceToHost);
		
		// free device arrays
		cudaFree(rho_d);
		cudaFree(elems_d);
		cudaEventSynchronize(stop);
		float ms = 0;
		cudaEventElapsedTime(&ms, start, stop);
		printf("milliseconds: %f\n", ms);
		printf("Testing compilation\n");
		return;
	}
}
/*
		printf("rho_h: %f\n", *rho_h);
		printf("mass: %f\n", elems_h->mass);
		printf("pos 0 0: %f\n", elems_h->pos[0].xyz[0]);
		printf("pos 0 1: %f\n", elems_h->pos[0].xyz[1]);
		printf("pos 0 2: %f\n", elems_h->pos[0].xyz[2]);

		printf("rho_h: %f\n", *rho_h);
		printf("mass: %f\n", elems_h->mass);
		printf("pos 0 0: %f\n", elems_h->pos[0].xyz[0]);
		printf("pos 0 1: %f\n", elems_h->pos[0].xyz[1]);
		printf("pos 0 2: %f\n", elems_h->pos[0].xyz[2]);
*/