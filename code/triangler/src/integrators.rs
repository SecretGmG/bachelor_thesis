use crate::{
    integrands::Integrand, parametrization::Parametrization, vectors::Vec3, IntegrationResult,
    Logger,
};
use rand::rng;
use rayon::prelude::*;
use symbolica::numerical_integration::{ContinuousGrid, Sample};

pub(crate) trait Integrator: Sync + Send {
    fn integrate<F: Integrand, P: Parametrization>(
        &self,
        integrand: &F,
        param: &P,
        logger: &Logger,
    ) -> IntegrationResult;
}

pub(crate) struct MonteCarlo {}
impl Integrator for MonteCarlo {
    fn integrate<F: Integrand, P: Parametrization>(
        &self,
        integrand: &F,
        param: &P,
        logger: &Logger,
    ) -> IntegrationResult {
        todo!()
    }
}
pub(crate) struct Vegas {
    epochs: usize,
    iters_per_epoch: usize,
}
impl Vegas {
    pub fn new(epochs: usize, iters_per_epoch: usize) -> Self {
        Self {
            epochs,
            iters_per_epoch,
        }
    }
}
impl Integrator for Vegas {
    fn integrate<T: Integrand, P: Parametrization>(
        &self,
        integrand: &T,
        param: &P,
        logger: &Logger,
    ) -> IntegrationResult {
        let mut grid = ContinuousGrid::new(3, 50, 1000, None, false);

        let mut rng = rand::rng();

        let mut sample = Sample::new();
        for _ in 0..self.epochs {
            for _ in 0..self.iters_per_epoch {
                grid.sample(&mut rng, &mut sample);

                if let Sample::Continuous(_wgt, xs) = &sample {
                    let xs = Vec3::new(xs[0], xs[1], xs[2]);
                    let (k, jac) = param.transform(xs);
                    let res = integrand.evaluate(k);
                    grid.add_training_sample(&sample, jac * res).unwrap();
                }
            }

            grid.update(1.0);

            let (mean, err, _) = grid.accumulator.get_live_estimate();
            logger.info(format!("mean: {mean}, err: {err}"));
        }
        let (mean, err, _) = grid.accumulator.get_live_estimate();
        IntegrationResult { mean, err }
    }
}
pub(crate) struct VegasMulti {
    epochs: usize,
    iters_per_epoch: usize,
    batches: usize,
}
impl VegasMulti {
    pub fn new(epochs: usize, iters_per_epoch: usize, batches: usize) -> Self {
        Self {
            epochs,
            iters_per_epoch,
            batches,
        }
    }
}
impl Integrator for VegasMulti {
    fn integrate<T: Integrand, P: Parametrization>(
        &self,
        integrand: &T,
        param: &P,
        logger: &Logger,
    ) -> IntegrationResult {
        let mut grid = ContinuousGrid::new(3, 50, 1000, None, false);
        for _ in 0..self.epochs {
            let subgrids: Vec<ContinuousGrid<f64>> = (0..self.batches)
                .into_par_iter()
                .map(|_| {
                    let mut grid_clone = grid.clone();
                    let mut rng = rng();
                    let mut sample = Sample::new();

                    for _ in 0..self.iters_per_epoch / self.batches {
                        grid_clone.sample(&mut rng, &mut sample);

                        if let Sample::Continuous(_wgt, xs) = &sample {
                            let xs_vec = Vec3::new(xs[0], xs[1], xs[2]);
                            let (k, jac) = param.transform(xs_vec);
                            let res = integrand.evaluate(k);
                            grid_clone.add_training_sample(&sample, jac * res).unwrap();
                        }
                    }

                    grid_clone
                })
                .collect();
            subgrids
                .iter()
                .for_each(|subgrid| grid.merge(subgrid).unwrap());
            grid.update(1.0);

            let (mean, err, _) = grid.accumulator.get_live_estimate();
            logger.info(format!("mean: {mean}, err: {err}"));
        }
        let (mean, err, _) = grid.accumulator.get_live_estimate();
        IntegrationResult { mean, err }
    }
}
