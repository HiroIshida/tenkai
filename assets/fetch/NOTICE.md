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
and simulator settings, and normalizes XML formatting. tenkai changes only the
header comment to describe this bundle. No meshes or binaries are included.

Original upstream URDF SHA-256, recorded by tenkai2:
`42c01b4faf7cbffdc92d2b1ecd655d02f2f33aa4df9fe4fc310a5661b0628b83`.

## Collision configuration

`collision.yaml` is copied unchanged from tenkai2's bundled Fetch assets,
originally `src/plainmp/conf/fetch.yaml` in
[HiroIshida/plainmp, commit c1c4db05433be2e1eafb6c634be3666a6aa18394](https://github.com/HiroIshida/plainmp/blob/c1c4db05433be2e1eafb6c634be3666a6aa18394/src/plainmp/conf/fetch.yaml).
Copyright (c) 2024 Hirokazu Ishida. It is distributed under BSD 3-Clause;
see [LICENSE-BSD3.txt](LICENSE-BSD3.txt).
Tenkai removed the machine-specific URDF path and added attribution comments.

tenkai currently reads all collision spheres for environment checks. Self-collision
settings and default/control joint settings in this file are not applied.
