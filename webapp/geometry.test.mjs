// Node built-in test runner:  node --test webapp/geometry.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { scDxlIntersection } from "./geometry.js";

// A vertical SC line at Y=0, running Z=-5..5 on a flat surface (X=5), seed at Z=0.
// The DXL is a horizontal line at Z=2. They cross at (Y=0, Z=2).
function verticalScLine() {
  const y = [], z = [], x = [];
  for (let zz = -5; zz <= 5; zz += 1) { y.push(0); z.push(zz); x.push(5); }
  const seed_index = z.indexOf(0);
  return { x, y, z, seed_index };
}

test("finds the crossing point", () => {
  const sc = verticalScLine();
  const dxl = { x: [5, 5], y: [-4, 4], z: [2, 2] };
  const r = scDxlIntersection(sc, dxl);
  assert.ok(r);
  assert.ok(Math.abs(r.y - 0) < 1e-9);
  assert.ok(Math.abs(r.z - 2) < 1e-9);
});

test("distance is 3D arc length from seed to crossing", () => {
  const sc = verticalScLine();       // flat X=5, so 3D length == |dZ|
  const dxl = { x: [5, 5], y: [-4, 4], z: [2, 2] };
  const r = scDxlIntersection(sc, dxl);
  assert.ok(Math.abs(r.dist_re - 2.0) < 1e-9);          // Z from 0 to 2
  assert.ok(Math.abs(r.dist_km - 2.0 * 6371.0) < 1e-6);
});

test("3D distance accounts for X curvature", () => {
  // Same YZ path but X ramps by 1 per step -> each unit-Z segment has
  // 3D length sqrt(1^2 + 1^2) = sqrt(2). Seed at Z=0, crossing at Z=2 -> 2 segs.
  const y = [], z = [], x = [];
  for (let zz = -5; zz <= 5; zz += 1) { y.push(0); z.push(zz); x.push(5 + zz); }
  const sc = { x, y, z, seed_index: z.indexOf(0) };
  const dxl = { x: [5, 5], y: [-4, 4], z: [2, 2] };
  const r = scDxlIntersection(sc, dxl);
  assert.ok(Math.abs(r.dist_re - 2 * Math.SQRT2) < 1e-9);
});

test("picks the crossing nearest the spacecraft along the line", () => {
  const sc = verticalScLine();                 // seed at Z=0
  // Two horizontal DXL segments: one at Z=1 (near), one at Z=4 (far).
  const dxl = { x: [5, 5, null, 5, 5], y: [-4, 4, null, -4, 4], z: [1, 1, null, 4, 4] };
  const r = scDxlIntersection(sc, dxl);
  assert.ok(Math.abs(r.z - 1) < 1e-9);         // nearest crossing chosen
  assert.ok(Math.abs(r.dist_re - 1.0) < 1e-9);
});

test("returns null when there is no crossing", () => {
  const sc = verticalScLine();
  const dxl = { x: [5, 5], y: [10, 20], z: [2, 2] };   // Y range doesn't overlap
  assert.equal(scDxlIntersection(sc, dxl), null);
});

test("returns null on missing input", () => {
  assert.equal(scDxlIntersection(null, { y: [1, 2], z: [1, 2] }), null);
  assert.equal(scDxlIntersection(verticalScLine(), null), null);
});
