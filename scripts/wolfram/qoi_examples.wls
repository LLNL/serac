#!/usr/bin/env wolframscript

f[x_, y_, u_] := x^2 + Sin[y] + x u^3 ;
g[x_, y_, u_] := x - y + Cos[u y];
u = x^2;
(* from "star.mesh" *)
vertices = {{0,0},{1,0},{0.309017`,0.951057`},{1.30902`,0.951057`},{-0.809017`,0.587785`},{-0.5`,1.53884`},{-0.809017`,-0.587785`},{-1.61803`,0},{0.309017`,-0.951057`},{-0.5`,-1.53884`},{1.30902`,-0.951057`},{0.5`,0},{1.15451`,0.475529`},{0.809019`,0.951057`},{0.154508`,0.475529`},{-0.0954915`,1.24495`},{-0.654508`,1.06331`},{-0.404508`,0.293893`},{-1.21352`,0.293893`},{-1.21352`,-0.293892`},{-0.404508`,-0.293893`},{-0.654508`,-1.06331`},{-0.0954915`,-1.24495`},{0.154508`,-0.475529`},{0.809019`,-0.951057`},{1.15451`,-0.475529`},{0.654509`,0.475529`},{-0.25`,0.769421`},{-0.809016`,0},{-0.25`,-0.76942`},{0.654509`,-0.475529`}};
boundarySegments={2,26,11,25,9,23,10,22,7,20,8,19,5,17,6,16,3,14,4,13};
domain = Polygon[vertices[[boundarySegments]]];
boundary = RegionBoundary[domain];
Print[NIntegrate[f[x,y,u],{x,y}\[Element]domain]+NIntegrate[g[x,y,u],{x,y}\[Element]boundary]]

(*
Graphics[{
  Arrow[vertices[[boundarySegments]]] & /@ boundarySegments,
  Red,PointSize[0.02],
  Point /@ vertices
}]
*)

(* beam-hex.mesh *)
domain = Cuboid[{0, 0,0}, {8, 1, 1}];
boundary = RegionBoundary[domain];
Print[NIntegrate[f[x,y,u],{x,y,z}\[Element]domain]+NIntegrate[g[x,y,u],{x,y,z}\[Element]boundary]]
(* RegionPlot3D[domain] *)
