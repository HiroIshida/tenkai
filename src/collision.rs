//! Robot collision spheres loaded from tenkai2/plainmp YAML.
//!
//! Only `collision_spheres: { link_name: { spheres: [[x, y, z, radius], ...] } }`
//! is read. Control/default joint settings and self-collision settings are not applied;
//! all listed spheres participate in environment checks, including links marked
//! `only_self_collision_links`. Joint inputs still follow `FkDag::joint_names()`.

use std::collections::BTreeMap;
use std::fmt;

use serde::Deserialize;

use crate::fk::{CentersError, FkDag, SphereCenter};

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: SphereCenter,
    /// Nonnegative radius in meters; zero represents a point.
    pub radius: f64,
}

impl Sphere {
    pub(crate) fn validate(&self, fk: &FkDag, sphere_index: usize) -> Result<(), Error> {
        self.center
            .validate(fk, sphere_index)
            .map_err(Error::Center)?;
        if !self.radius.is_finite() || self.radius < 0.0 {
            return Err(Error::InvalidRadius { sphere_index });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum Error {
    Yaml(Box<serde_saphyr::Error>),
    UnknownLink(String),
    Center(CentersError),
    InvalidRadius { sphere_index: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Yaml(error) => write!(f, "collision sphere YAML: {error}"),
            Self::UnknownLink(name) => {
                write!(f, "collision spheres reference unknown link {name:?}")
            }
            Self::Center(error) => error.fmt(f),
            Self::InvalidRadius { sphere_index } => {
                write!(
                    f,
                    "sphere {sphere_index}: radius must be finite and nonnegative"
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Yaml(error) => Some(error.as_ref()),
            Self::Center(error) => Some(error),
            _ => None,
        }
    }
}

/// Resolves link names against `fk`. Output follows FK link order, then YAML sphere order.
/// A missing `collision_spheres` field is an error; an explicit empty map is valid.
pub fn read_spheres(yaml: &str, fk: &FkDag) -> Result<Vec<Sphere>, Error> {
    let mut config: Config =
        serde_saphyr::from_str(yaml).map_err(|error| Error::Yaml(Box::new(error)))?;
    let mut spheres = Vec::new();
    for (link_index, pose) in fk.link_poses().iter().enumerate() {
        if let Some(group) = config.collision_spheres.remove(&pose.name) {
            for [x, y, z, radius] in group.spheres {
                let sphere = Sphere {
                    center: SphereCenter {
                        link_index,
                        position: [x, y, z],
                    },
                    radius,
                };
                sphere.validate(fk, spheres.len())?;
                spheres.push(sphere);
            }
        }
    }
    if let Some((name, _)) = config.collision_spheres.into_iter().next() {
        return Err(Error::UnknownLink(name));
    }
    Ok(spheres)
}

#[derive(Deserialize)]
struct Config {
    collision_spheres: BTreeMap<String, Group>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Group {
    spheres: Vec<[f64; 4]>,
}
