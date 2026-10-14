#!/usr/bin/env python3
"""
viewer_planes.py -- the interactive 3D viewer, plus two translucent light-red planes standing on the
bowling-end and batting-end POPPING CREASES, with the ball's release / impact / crossing points marked
on them and their height above the ground written next to each marker.

Put this file next to ball3d.py (it re-uses its viewer) and run:

    python viewer_planes.py --track track3d_83.json --out viewer_planes.html

or add the planes to a viewer you already made:

    python viewer_planes.py --html viewer.html --out viewer_planes.html

What is drawn (works in every view: camera, bowler end, side, top, batter end, follow):
  * two light-red transparent planes across the pitch on the popping creases, each with a red outline
    and a red strip on the ground (so the plane is still visible when seen edge-on from the top/side)
  * RELEASE point (green) shown on the bowling-crease plane and IMPACT point (red) shown on the
    batting-crease plane, each with a height label ("release - height 2.06 m") and a white drop line
    to the ground. If the real release/impact is not exactly on the plane (e.g. the ball was released
    1 m after the crease) a dashed line joins the real point to its spot on the plane.
  * a white marker + height label wherever the ball's path actually passes THROUGH a plane
  * a checkbox "crease planes + heights" in the viewer to switch it all on/off
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ball3d as b3                                                    # noqa: E402

MARKER = "// playback"             # the planes are injected just before this line of the viewer's script

PLANES_JS = r"""
// ===== crease planes (light red, transparent) + release / impact / crossing heights =====
(function(){
  const XB = PIT.popping, XT = PIT.len - PIT.popping;                  // bowling / batting popping crease (world X)
  const WID = PIT.half_w*2, HGT = Math.max(3.0, Math.max.apply(null, D.z) + 0.6);
  const RED = 0xff5a5a, RED2 = 0xff2b2b;
  const group = new THREE.Group(); scene.add(group);
  const labels = [];

  function label(p3, html, color, dy){
    const el = document.createElement('div'); el.className = 'lab'; el.innerHTML = html;
    el.style.borderLeft = '3px solid ' + color; el.style.textAlign = 'left';
    document.getElementById('labels').appendChild(el); labels.push({pos: p3.clone(), el, dy: dy || 0});
  }
  function dot(p3, color, r){
    const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), new THREE.MeshBasicMaterial({color, depthTest:false}));
    m.position.copy(p3); m.renderOrder = 10; group.add(m);
  }
  function dropLine(p3){
    const g = new THREE.BufferGeometry().setFromPoints([p3, new THREE.Vector3(p3.x, 0.004, p3.z)]);
    group.add(new THREE.Line(g, new THREE.LineBasicMaterial({color:0xffffff, transparent:true, opacity:0.9})));
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.085, 24), new THREE.MeshBasicMaterial({color:0xffffff, side:THREE.DoubleSide, transparent:true, opacity:0.95}));
    ring.rotation.x = -Math.PI/2; ring.position.set(p3.x, 0.008, p3.z); group.add(ring);
  }
  function dashed(a, b, color){
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    const l = new THREE.Line(g, new THREE.LineDashedMaterial({color, dashSize:0.12, gapSize:0.08}));
    l.computeLineDistances(); group.add(l);
  }

  // ---- the two planes
  const planeMat = new THREE.MeshBasicMaterial({color:RED, transparent:true, opacity:0.22, side:THREE.DoubleSide, depthWrite:false});
  function crease(X, name){
    const pl = new THREE.Mesh(new THREE.PlaneGeometry(WID, HGT), planeMat);
    pl.rotation.y = Math.PI/2; pl.position.set(X, HGT/2, 0); group.add(pl);          // spans the pitch width and up in the air
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(pl.geometry), new THREE.LineBasicMaterial({color:RED2}));
    edges.rotation.copy(pl.rotation); edges.position.copy(pl.position); group.add(edges);
    const band = new THREE.Mesh(new THREE.PlaneGeometry(0.12, WID), new THREE.MeshBasicMaterial({color:RED2, transparent:true, opacity:0.8, depthWrite:false}));
    band.rotation.x = -Math.PI/2; band.position.set(X, 0.006, 0); group.add(band);   // red strip on the ground = the crease line
    label(new THREE.Vector3(X, HGT, 0), name, '#ff5a5a', 0);
  }
  crease(XB, 'bowling crease plane');
  crease(XT, 'batting crease plane');

  // ---- where the path passes through a plane (null if it never does)
  function crossing(X){
    for(let i = 0; i < N-1; i++){
      const a = D.x[i], b = D.x[i+1];
      if(a !== b && (a-X)*(b-X) <= 0){
        const f = (X-a)/(b-a);
        return {y: D.y[i]+f*(D.y[i+1]-D.y[i]), z: D.z[i]+f*(D.z[i+1]-D.z[i])};
      }
    }
    return null;
  }

  // ---- release on the bowling plane, impact on the batting plane
  function pointOnPlane(key, X, color, hex, name){
    if(!M[key]) return;
    const p = M[key].pos, real = w2t(p[0], p[1], p[2]), onPlane = w2t(X, p[1], p[2]);
    dot(onPlane, hex, 0.07); dropLine(onPlane);
    let extra = '';
    if(Math.abs(p[0]-X) > 0.05){
      dashed(real, onPlane, hex);                                                      // joins the real point to its spot on the plane
      extra = ' <span style="opacity:.7">(real X=' + p[0].toFixed(1) + ' m)</span>';
    }
    label(onPlane, name + ' &middot; <b>' + p[2].toFixed(2) + ' m</b>' + extra, color, 0);
  }
  pointOnPlane('release', XB, '#2ecc71', 0x2ecc71, 'release');
  pointOnPlane('impact',  XT, '#ff6b6b', 0xe74c3c, 'impact');

  // ---- where the ball really passes through each plane
  [[XB, 'bowling'], [XT, 'batting']].forEach(function(pair){
    const X = pair[0], nm = pair[1], c = crossing(X);
    if(!c) return;
    const key = (X === XB) ? 'release' : 'impact';
    if(M[key] && Math.abs(M[key].pos[0]-X) <= 0.05) return;                         // same spot as the release/impact marker
    const p = w2t(X, c.y, c.z);
    dot(p, 0xffffff, 0.055); dropLine(p);
    label(p, 'passes ' + nm + ' crease &middot; <b>' + c.z.toFixed(2) + ' m</b>', '#ffffff', 40);
  });

  // ---- on/off switch + label placement (own loop, so it can hide the labels too)
  const row = document.createElement('div'); row.className = 'row';
  row.innerHTML = '<label><input type="checkbox" id="planes" checked> crease planes + heights</label>';
  document.getElementById('grow').closest('.card').appendChild(row);
  document.getElementById('planes').onchange = function(e){ group.visible = e.target.checked; };
  function placeLabels(){
    const on = document.getElementById('planes').checked;
    labels.forEach(function(m){
      const v = m.pos.clone().project(camera);
      m.el.style.display = (on && v.z < 1) ? 'block' : 'none';
      m.el.style.left = ((v.x*0.5+0.5)*innerWidth) + 'px';
      m.el.style.top  = ((-v.y*0.5+0.5)*innerHeight + m.dy) + 'px';
    });
    requestAnimationFrame(placeLabels);
  }
  requestAnimationFrame(placeLabels);
})();

"""


def add_planes(html):
    if "crease planes (light red" in html:
        return html
    if MARKER not in html:
        raise SystemExit("this viewer html is not the ball3d.py viewer (marker %r not found)" % MARKER)
    return html.replace(MARKER, PLANES_JS + MARKER, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--track", help="track3d.json from `ball3d.py fit`")
    src.add_argument("--html", help="an existing viewer.html made by `ball3d.py viewer`")
    ap.add_argument("--out", default="viewer_planes.html")
    ap.add_argument("--three-src", default=b3.THREE_CDN, help="URL/path of three.min.js (r128), for offline use")
    a = ap.parse_args()

    if a.track:
        track = json.load(open(a.track))
        html = b3.VIEWER_HTML.replace("__THREE__", a.three_src).replace("__DATA__", json.dumps(track))
    else:
        html = open(a.html).read()
    with open(a.out, "w") as fp:
        fp.write(add_planes(html))
    print("[viewer] wrote %s -- open it in a browser (tick/untick 'crease planes + heights' to compare)" % a.out)


if __name__ == "__main__":
    main()
