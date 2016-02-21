/*
 *
 *		test_helpers.c 
 *		
 *		Helper functions for r3d and r2d unit testing.
 *		
 *		Devon Powell
 *		31 August 2015
 *
 *		This program was prepared by Los Alamos National Security, LLC at Los Alamos National
 *		Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy (DOE). 
 *		All rights in the program are reserved by the DOE and Los Alamos National Security, LLC.  
 *		Permission is granted to the public to copy and use this software without charge, provided that 
 *		this Notice and any statement of authorship are reproduced on all copies.  Neither the U.S. 
 *		Government nor LANS makes any warranty, express or implied, or assumes any liability 
 *		or responsibility for the use of this software.
 *
 */

#include "test_helpers.h"


/**
 * Random number utilities
 */

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
	//return sqrt(-2.0*log(u1))*sin(TWOPI*u2);
}


/**
 *
 * 3D helper functions. 
 *
 */


// simple 3-vector operations
#define dot3(va, vb) (va.x*vb.x + va.y*vb.y + va.z*vb.z)
#define norm3(v) {					\
	r3d_real tmplen = sqrt(dot3(v, v));	\
	v.x /= (tmplen + 1.0e-299);		\
	v.y /= (tmplen + 1.0e-299);		\
	v.z /= (tmplen + 1.0e-299);		\
}


r3d_plane (*choptions_3d[6]) (r3d_poly* poly); // keep an array of these cutting options

r3d_plane thru_cent_3d(r3d_poly* poly) {
	// make a randomly-oriented plane passing through the centroid
	r3d_rvec3 centroid = get_centroid_3d(poly);
	r3d_plane plane;
	plane.n = rand_uvec_3d();
	plane.d = -dot3(plane.n, centroid);
	return plane;
}

r3d_plane thru_face_3d(r3d_poly* poly) {
	// make a plane coplanar with a face of the poly
	r3d_int v0 = rand_int(poly->nverts);
	r3d_int pnbr1 = rand_int(3);
	r3d_int pnbr2 = (pnbr1 + rand_int(2) + 1)%3;
	r3d_int v1 = poly->verts[v0].pnbrs[pnbr1];
	r3d_int v2 = poly->verts[v0].pnbrs[pnbr2];
	r3d_rvec3 p0 = poly->verts[v0].pos;
	r3d_rvec3 p1 = poly->verts[v1].pos;
	r3d_rvec3 p2 = poly->verts[v2].pos;
	return point_plane_3d(p0, p1, p2);
}


r3d_plane thru_edge_cent_3d(r3d_poly* poly) {
	// make a plane coplanar with an edge and the centroid of the poly 
	r3d_int v0 = rand_int(poly->nverts);
	r3d_int pnbr1 = rand_int(3);
	r3d_int v1 = poly->verts[v0].pnbrs[pnbr1];
	r3d_rvec3 p0 = poly->verts[v0].pos;
	r3d_rvec3 p1 = poly->verts[v1].pos;
	r3d_rvec3 centroid = get_centroid_3d(poly);
	return point_plane_3d(p0, p1, centroid);
}

r3d_plane thru_edge_rand_3d(r3d_poly* poly) {
	// make a plane coplanar with an edge, but otherwise randomly oriented 
	r3d_int v0 = rand_int(poly->nverts);
	r3d_int pnbr1 = rand_int(3);
	r3d_int v1 = poly->verts[v0].pnbrs[pnbr1];
	r3d_rvec3 p0 = poly->verts[v0].pos;
	r3d_rvec3 p1 = poly->verts[v1].pos;
	return point_plane_3d(p0, p1, rand_uvec_3d());
}

r3d_plane thru_vert_cent_3d(r3d_poly* poly) {
	// make a plane coplanar with a vertex and the centroid, but otherwise randomly oriented 
	r3d_int v0 = rand_int(poly->nverts);
	r3d_rvec3 p0 = poly->verts[v0].pos;
	r3d_rvec3 centroid = get_centroid_3d(poly);
	return point_plane_3d(p0, centroid, rand_uvec_3d());
}

r3d_plane thru_vert_rand_3d(r3d_poly* poly) {
	// make a plane coplanar with a vertex, but otherwise randomly oriented 
	r3d_int v0 = rand_int(poly->nverts);
	r3d_rvec3 p0 = poly->verts[v0].pos;
	r3d_plane plane;
	plane.n = rand_uvec_3d();
	plane.d = -dot3(plane.n, p0);
	return plane;
}

r3d_plane point_plane_3d(r3d_rvec3 p0, r3d_rvec3 p1, r3d_rvec3 p2) {
	// generate a plane passing through the three points given
	r3d_plane plane;
	plane.n.x = (p1.y - p0.y)*(p2.z - p0.z) - (p1.z - p0.z)*(p2.y - p0.y);
	plane.n.y = (p1.z - p0.z)*(p2.x - p0.x) - (p1.x - p0.x)*(p2.z - p0.z);
	plane.n.z = (p1.x - p0.x)*(p2.y - p0.y) - (p1.y - p0.y)*(p2.x - p0.x);
	norm3(plane.n);
	plane.d = -dot3(plane.n, p0);
	return plane;
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

r3d_real rand_tet_3d(r3d_rvec3 verts[4], r3d_real minvol) {
	// generates a random tetrahedron with vertices on the unit sphere,
	// guaranteeing a volume of at least MIN_VOL (to avoid degenerate cases)
	r3d_int v;
	r3d_rvec3 swp;
	r3d_real tetvol = 0.0;
	while(tetvol < minvol) {
		for(v = 0; v < 4; ++v) 
			verts[v] = rand_uvec_3d();
		tetvol = r3d_orient(verts);
		if(tetvol < 0.0) {
			swp = verts[2];
			verts[2] = verts[3];
			verts[3] = swp;
			tetvol = -tetvol;
		}
	}
	return tetvol;
}

r3d_rvec3 get_centroid_3d(r3d_poly* poly) {
	// get the "centroid" by averaging vertices
	r3d_rvec3 centroid;
	centroid.x = 0.0;
	centroid.y = 0.0;
	centroid.z = 0.0;
	r3d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		centroid.x += poly->verts[v].pos.x;
		centroid.x += poly->verts[v].pos.y;
		centroid.x += poly->verts[v].pos.z;
	}
	centroid.x /= poly->nverts;
	centroid.y /= poly->nverts;
	centroid.z /= poly->nverts;
	return centroid;
}