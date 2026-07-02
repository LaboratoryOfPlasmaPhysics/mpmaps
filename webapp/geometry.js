// Pure geometry helpers for the spacecraft field-line / DXL overlay.
// No DOM or Plotly dependency — importable by app.js and by node tests.

const RE_KM = 6371.0;

// Intersection parameter of segment A(p1->p2) with segment B(p3->p4), in the
// (Y,Z) plane. Returns { t, u, y, z } where t is the fraction along A and u
// the fraction along B, both in [0,1], or null if the segments don't cross.
function segIntersect(p1, p2, p3, p4) {
  const r_y = p2.y - p1.y, r_z = p2.z - p1.z;
  const s_y = p4.y - p3.y, s_z = p4.z - p3.z;
  const denom = r_y * s_z - r_z * s_y;
  if (Math.abs(denom) < 1e-12) return null;   // parallel / degenerate
  const qp_y = p3.y - p1.y, qp_z = p3.z - p1.z;
  const t = (qp_y * s_z - qp_z * s_y) / denom;
  const u = (qp_y * r_z - qp_z * r_y) / denom;
  if (t < 0 || t > 1 || u < 0 || u > 1) return null;
  return { t, u, y: p1.y + t * r_y, z: p1.z + t * r_z };
}

// Cumulative 3D arc length along a polyline given as parallel x/y/z arrays.
// s[k] = arc length from vertex 0 to vertex k.
function cumulativeArcLength(x, y, z) {
  const s = new Array(x.length);
  s[0] = 0;
  for (let k = 1; k < x.length; k++) {
    const dx = x[k] - x[k - 1], dy = y[k] - y[k - 1], dz = z[k] - z[k - 1];
    s[k] = s[k - 1] + Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return s;
}

// Find where the spacecraft field line crosses the dominant X-line and the
// 3D distance from the spacecraft (the seed vertex) to that crossing, measured
// along the field line.
//
//   scLine:   { x:[], y:[], z:[], seed_index }
//   dxlCurve: { x:[], y:[], z:[] }   (may contain nulls marking gaps)
//
// Returns { y, z, x, dist_re, dist_km } for the crossing nearest the
// spacecraft along the line, or null if there is no crossing / bad input.
export function scDxlIntersection(scLine, dxlCurve) {
  if (!scLine || !dxlCurve) return null;
  const { x, y, z, seed_index } = scLine;
  if (!x || x.length < 2 || seed_index == null) return null;

  const dx = dxlCurve.x, dy = dxlCurve.y, dz = dxlCurve.z;
  if (!dy || dy.length < 2) return null;

  const s = cumulativeArcLength(x, y, z);
  const sSeed = s[seed_index];

  let best = null;   // { dist, y, z, x }
  for (let i = 0; i < x.length - 1; i++) {
    const segLen = s[i + 1] - s[i];
    const a1 = { y: y[i], z: z[i] };
    const a2 = { y: y[i + 1], z: z[i + 1] };
    for (let j = 0; j < dy.length - 1; j++) {
      // Skip DXL segments touching a null gap.
      if (dy[j] == null || dz[j] == null || dy[j + 1] == null || dz[j + 1] == null)
        continue;
      const hit = segIntersect(
        a1, a2,
        { y: dy[j], z: dz[j] },
        { y: dy[j + 1], z: dz[j + 1] },
      );
      if (!hit) continue;
      // Arc length of the crossing along the SC line, then distance from seed.
      const sHit = s[i] + hit.t * segLen;
      const dist = Math.abs(sHit - sSeed);
      if (best === null || dist < best.dist) {
        const xHit = x[i] + hit.t * (x[i + 1] - x[i]);
        best = { dist, y: hit.y, z: hit.z, x: xHit };
      }
    }
  }
  if (best === null) return null;
  return {
    y: best.y, z: best.z, x: best.x,
    dist_re: best.dist,
    dist_km: best.dist * RE_KM,
  };
}
