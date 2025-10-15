pub(crate) trait Integrator: Sync+Send{
    fn integrate(&self) -> IntegrationResult;
}

pub(crate) struct MonteCarlo{

}
impl Integrator for MonteCarlo {
    fn integrate(&self) -> IntegrationResult {
        todo!()
    }
}

pub(crate) struct IntegrationResult{

}