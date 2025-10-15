use crate::{integrands::Integrand, parametrization::Parametrization, vectors::Vec3, IntegrationResult, Logger};
use pyo3::{Py, PyAny, Python};
use symbolica::numerical_integration::{ContinuousGrid, Grid, Sample};

pub(crate) trait Integrator: Sync + Send {
    fn integrate<F:Integrand, P: Parametrization>(&self, integrand:&F, param:&P, logger: &Logger) -> IntegrationResult;
}

pub(crate) struct MonteCarlo {}
impl Integrator for MonteCarlo {
    fn integrate<F:Integrand, P: Parametrization>(&self, integrand:&F, param:&P, logger: &Logger) -> IntegrationResult {
        todo!()
    }
}
pub(crate) struct Vegas {
    epochs : u64,
    iters_per_epoch : u64

}
impl Vegas{
    pub fn new(epochs: u64, iters_per_epoch: u64)->Self{
        Self{ epochs, iters_per_epoch}
    }
}
impl Integrator for Vegas{
    fn integrate<T: Integrand, P:Parametrization>(&self, integrand: &T, param:&P, logger: &Logger) -> IntegrationResult {
        let mut grid = Grid::Continuous(ContinuousGrid::new(3, 10, 1000, None, false));

        let mut rng = rand::rng();

        let mut sample = Sample::new();
        for _ in 0..self.epochs {
            for _ in 0..self.iters_per_epoch {
                grid.sample(&mut rng, &mut sample);

                if let Sample::Continuous(_wgt, xs) = &sample {
                    let xs = Vec3::new(xs[0], xs[1],  xs[2]);
                    let (k, jac) = param.transform(xs);
                    let res = integrand.evaluate(k);
                    grid.add_training_sample(&sample, jac*res).unwrap();
                }
            }

            grid.update(0.0,1.0);

            let stats = grid.get_statistics();
            logger.info(format!("{stats:?}"));
        }
        let stats = grid.get_statistics();
        IntegrationResult { mean: stats.avg, err: stats.err }
    }
}
