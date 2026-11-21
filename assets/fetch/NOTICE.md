# Fetch assets

`fetch.urdf` is distributed under **CC BY-NC-SA 4.0** (Attribution,
NonCommercial, ShareAlike). See the full text in
[LICENSE-CC-BY-NC-SA-4.0.txt](LICENSE-CC-BY-NC-SA-4.0.txt).
This license applies to the model asset separately from tenkai's source code.

Authors: Michael Ferguson and Melonee Wise (Fetch Robotics).
The original model is
[`fetch_description/robots/fetch.urdf`](https://github.com/fetchrobotics/fetch_ros/blob/d22b98f6c21577895f4d2e3a8bd54e329083732b/fetch_description/robots/fetch.urdf)
at commit `d22b98f6c21577895f4d2e3a8bd54e329083732b`; its
[package declaration](https://github.com/fetchrobotics/fetch_ros/blob/d22b98f6c21577895f4d2e3a8bd54e329083732b/fetch_description/package.xml)
specifies this license.

Copied from tenkai2's `src/tenkai/assets/fetch/fetch.urdf`. The tenkai adaptation
retains link names and joint elements, including origins, axes, and limits;
it removes visual/collision geometry, inertial data, materials, transmissions,
and simulator settings, and normalizes XML formatting. tenkai also adds VAMP's
fixed `torso_lift_link_collision_2` child of `torso_lift_link` so its front sphere
can retain separate self-collision pairs. Existing joint origins and limits are
unchanged. No meshes or binaries are included.

Original upstream URDF SHA-256, recorded by tenkai2:
`42c01b4faf7cbffdc92d2b1ecd655d02f2f33aa4df9fe4fc310a5661b0628b83`.

## Collision configuration

`collision.yaml` contains VAMP's 111 Fetch collision spheres and 48 self-collision
link pairs (2,586 sphere pairs), adapted from
[KavrakiLab/vamp, commit 681813351b745f7b5732761a026b06a97c494a95](https://github.com/KavrakiLab/vamp/tree/681813351b745f7b5732761a026b06a97c494a95).
VAMP is distributed under Apache 2.0; see
[LICENSE-APACHE-2.0.txt](LICENSE-APACHE-2.0.txt).

Local sphere centers and radii were extracted from
[`resources/fetch/fetch_spherized.urdf`](https://github.com/KavrakiLab/vamp/blob/681813351b745f7b5732761a026b06a97c494a95/resources/fetch/fetch_spherized.urdf)
and checked against `Fetch::sphere_fk` in
[`src/impl/vamp/robots/fetch.hh`](https://github.com/KavrakiLab/vamp/blob/681813351b745f7b5732761a026b06a97c494a95/src/impl/vamp/robots/fetch.hh).
Link pairs were recovered from the sphere indices in `Fetch::fkcc_debug`;
each pair covers the full Cartesian product of its links' spheres.

The adaptation excludes `base_link` from environment checks because its spheres
extend below the ground plane in the fixed-base example. All 111 spheres remain
available for self collision; the other 97 also check external obstacles.
No separate body collision primitives are needed.

Sphere centers remain in their original link frames. VAMP fixes the fingers open
at 0.05 m; tenkai retains the original movable finger joints, which stay at zero
during planning. `tests/fixtures/fetch_vamp.json` records sphere FK from VAMP's
SIMD C++ implementation with the fingers open, plus self-collision results from
its sphere pairs. Tests set those finger joints to 0.05 m for comparison.

The control joint order is retained from the previous plainmp-based configuration,
originally `src/plainmp/conf/fetch.yaml` in
[HiroIshida/plainmp, commit c1c4db05433be2e1eafb6c634be3666a6aa18394](https://github.com/HiroIshida/plainmp/blob/c1c4db05433be2e1eafb6c634be3666a6aa18394/src/plainmp/conf/fetch.yaml).
Copyright (c) 2024 Hirokazu Ishida, distributed under BSD 3-Clause;
see [LICENSE-BSD3.txt](LICENSE-BSD3.txt). The unused default joint positions and
body primitives from that configuration have been removed.
