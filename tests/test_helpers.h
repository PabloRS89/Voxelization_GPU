/*
 *
 *		test_helpers.h 
 *		
 *		Helper function declarations for r3d unit testing.
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

#ifndef _TEST_HELPERS_H_
#define _TEST_HELPERS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "r3d.h"

/*
 * Random number utilities.
 */

// random number generators
int rand_int(int N); // random integers from 0 (incl) to N (excl)
double rand_uniform(); // random uniform in (0, 1)
double rand_normal(); // random normal


/*
 * 3D geometry utilities.
 */

// functions for generating different clip plane orientations around a polyhedron
r3d_plane thru_cent_3d(r3d_poly* poly);
r3d_plane thru_face_3d(r3d_poly* poly);
r3d_plane thru_edge_cent_3d(r3d_poly* poly);
r3d_plane thru_edge_rand_3d(r3d_poly* poly);
r3d_plane thru_vert_cent_3d(r3d_poly* poly);
r3d_plane thru_vert_rand_3d(r3d_poly* poly);
extern r3d_plane (*choptions_3d[6]) (r3d_poly* poly); // keep an array of these cutting options

// get the centroid of a poly
r3d_rvec3 get_centroid_3d(r3d_poly* poly);

// generate a plane through the three points
r3d_plane point_plane_3d(r3d_rvec3 p0, r3d_rvec3 p1, r3d_rvec3 p2);

// randomly oriented (isotropic) unit vector
r3d_rvec3 rand_uvec_3d(); 

// random tet with verts on the unit sphere
r3d_real rand_tet_3d(r3d_rvec3 verts[4], r3d_real minvol); 

#endif // _TEST_HELPERS_H_
