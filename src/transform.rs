//! Pose operations expanded into scalar DAG nodes.

use crate::dag::{Dag, NodeId};

/// Components belong to one DAG; rotation is a unit quaternion in xyzw order.
#[derive(Clone, Copy)]
pub(crate) struct PoseNodes {
    pub(crate) translation: [NodeId; 3],
    pub(crate) rotation: [NodeId; 4],
}

impl PoseNodes {
    pub(crate) fn identity(dag: &mut Dag) -> Self {
        let zero = dag.constant(0.0);
        let one = dag.constant(1.0);
        Self {
            translation: [zero; 3],
            rotation: [zero, zero, zero, one],
        }
    }

    pub(crate) fn from_origin(dag: &mut Dag, origin: &crate::urdf::Pose) -> Self {
        // URDF origins are fixed, so their RPY conversion needs no runtime DAG nodes.
        Self {
            translation: origin.xyz.0.map(|v| dag.constant(v)),
            rotation: rpy_quaternion(origin.rpy.0).map(|v| dag.constant(v)),
        }
    }

    pub(crate) fn compose(self, dag: &mut Dag, local: Self) -> Self {
        Self {
            translation: self.transform_point(dag, local.translation),
            rotation: quaternion_product(dag, self.rotation, local.rotation),
        }
    }

    pub(crate) fn transform_point(self, dag: &mut Dag, point: [NodeId; 3]) -> [NodeId; 3] {
        let offset = rotate_vector(dag, self.rotation, point);
        [0, 1, 2].map(|i| dag.add(self.translation[i], offset[i]))
    }

    pub(crate) fn rotation_matrix(self, dag: &mut Dag) -> [[NodeId; 3]; 3] {
        let [x, y, z, w] = self.rotation;
        let one = dag.constant(1.0);
        let two = dag.constant(2.0);
        let [xx, yy, zz, xy, xz, yz, wx, wy, wz] = [
            (x, x),
            (y, y),
            (z, z),
            (x, y),
            (x, z),
            (y, z),
            (w, x),
            (w, y),
            (w, z),
        ]
        .map(|(a, b)| {
            let product = dag.mul(a, b);
            dag.mul(two, product)
        });
        let diagonal = [(yy, zz), (xx, zz), (xx, yy)].map(|(a, b)| {
            let sum = dag.add(a, b);
            dag.sub(one, sum)
        });
        [
            [diagonal[0], dag.sub(xy, wz), dag.add(xz, wy)],
            [dag.add(xy, wz), diagonal[1], dag.sub(yz, wx)],
            [dag.sub(xz, wy), dag.add(yz, wx), diagonal[2]],
        ]
    }

    pub(crate) fn transform_point_by_matrix(
        self,
        dag: &mut Dag,
        point: [NodeId; 3],
        rotation: [[NodeId; 3]; 3],
    ) -> [NodeId; 3] {
        [0, 1, 2].map(|i| {
            let [x, y, z] = [0, 1, 2].map(|j| dag.mul(rotation[i][j], point[j]));
            let xy = dag.add(x, y);
            let offset = dag.add(xy, z);
            dag.add(self.translation[i], offset)
        })
    }
}

fn rpy_quaternion([roll, pitch, yaw]: [f64; 3]) -> [f64; 4] {
    let (sr, cr) = (roll * 0.5).sin_cos();
    let (sp, cp) = (pitch * 0.5).sin_cos();
    let (sy, cy) = (yaw * 0.5).sin_cos();
    [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]
}

fn cross(dag: &mut Dag, a: [NodeId; 3], b: [NodeId; 3]) -> [NodeId; 3] {
    [[1, 2], [2, 0], [0, 1]].map(|[j, k]| {
        let positive = dag.mul(a[j], b[k]);
        let negative = dag.mul(a[k], b[j]);
        dag.sub(positive, negative)
    })
}

pub(crate) fn quaternion_product(dag: &mut Dag, a: [NodeId; 4], b: [NodeId; 4]) -> [NodeId; 4] {
    let cross = cross(dag, [a[0], a[1], a[2]], [b[0], b[1], b[2]]);
    let [x, y, z] = [0, 1, 2].map(|i| {
        let left = dag.mul(a[3], b[i]);
        let right = dag.mul(b[3], a[i]);
        let sum = dag.add(left, right);
        dag.add(sum, cross[i])
    });
    let products = [0, 1, 2, 3].map(|i| dag.mul(a[i], b[i]));
    let sum = dag.add(products[0], products[1]);
    let dot = dag.add(sum, products[2]);
    let w = dag.sub(products[3], dot);
    [x, y, z, w]
}

fn rotate_vector(dag: &mut Dag, q: [NodeId; 4], v: [NodeId; 3]) -> [NodeId; 3] {
    let imaginary = [q[0], q[1], q[2]];
    let two = dag.constant(2.0);
    let twice_cross = cross(dag, imaginary, v).map(|value| dag.mul(two, value));
    let correction = cross(dag, imaginary, twice_cross);
    [0, 1, 2].map(|i| {
        let scaled = dag.mul(q[3], twice_cross[i]);
        let sum = dag.add(v[i], scaled);
        dag.add(sum, correction[i])
    })
}
