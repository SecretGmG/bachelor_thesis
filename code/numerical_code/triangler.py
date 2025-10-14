#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Callable, Iterator

from pprint import pprint, pformat
import argparse
import math
import random
import time
import logging
import multiprocessing
import copy
from enum import StrEnum

try:
    import vegas
except:
    pass

try:
    from symbolica import Sample, NumericalIntegrator
except:
    pass

try:
    from numerical_code import ltd_triangle
except:
    pass

from vectors import LorentzVector, Vector

class Colour(StrEnum):
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

logging.basicConfig(
    format=f'{Colour.GREEN}%(levelname)s{Colour.END} {Colour.BLUE}%(funcName)s l.%(lineno)d{Colour.END} {Colour.CYAN}t=%(asctime)s.%(msecs)03d{Colour.END} > %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)
logger = logging.getLogger('Triangler')

TOLERANCE: float = 1e-10

RESCALING: float = 10.

class TrianglerException(Exception):
    pass

def chunks(a_list: list[Any], n: int) -> Iterator[list[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(a_list), n):
        yield a_list[i:i + n]

class SymbolicaSample(object):
    def __init__(self, sample: Sample):
        self.c: list[float] = sample.c
        self.d: list[int] = sample.d

class IntegrationResult(object):

    def __init__(self, 
                 central_value: float, error: float, n_samples: int = 0, elapsed_time: float = 0.,
                 max_wgt: float | None = None, 
                 max_wgt_point: list[float] | None = None):
        self.n_samples = n_samples
        self.central_value = central_value
        self.error = error
        self.max_wgt = max_wgt
        self.max_wgt_point = max_wgt_point
        self.elapsed_time = elapsed_time

    def combine_with(self, other):
        """ Combine self statistics with all those of another IntegrationResult object."""
        self.n_samples += other.n_samples
        self.elapsed_time += other.elapsed_time
        self.central_value += other.central_value
        self.error += other.error
        if other.max_wgt is not None:
            if self.max_wgt is None or abs(self.max_wgt) > abs(other.max_wgt):
                self.max_wgt = other.max_wgt
                self.max_wgt_point = other.max_wgt_point

    def normalize(self):
        """ Normalize the statistics."""
        self.central_value /= self.n_samples
        self.error = math.sqrt(abs(self.error / self.n_samples - self.central_value**2)/self.n_samples)

    def str_report(self, target: float | None = None) -> str:

        if self.central_value == 0. or self.n_samples == 0:
            return 'No integration result available yet'

        # First printout sample and timing statitics
        report = [f'Integration result after {Colour.GREEN}{self.n_samples}{Colour.END} evaluations in {Colour.GREEN}{self.elapsed_time:.2f} CPU-s{Colour.END}']
        if self.elapsed_time > 0.:
            report[-1] += f' {Colour.BLUE}({1.0e6*self.elapsed_time/self.n_samples:.1f} µs / eval){Colour.END}'

        # Also indicate max weight encountered if provided
        if self.max_wgt is not None and self.max_wgt_point is not None:
            report.append(f"Max weight encountered = {self.max_wgt:.5e} at xs = [{' '.join(f'{x:.16e}' for x in self.max_wgt_point)}]")

        # Finally return information about current best estimate of the central value
        report.append(f'{Colour.GREEN}Central value{Colour.END} : {self.central_value:<+25.16e} +/- {self.error:<12.2e}')

        err_perc = self.error/self.central_value*100
        if err_perc < 1.:
            report[-1] += f' ({Colour.GREEN}{err_perc:.3f}%{Colour.END})'
        else:
            report[-1] += f' ({Colour.RED}{err_perc:.3f}%{Colour.END})'

        # Also indicate distance to target if specified
        if target is not None and target != 0.:
            report.append(f'    vs target : {target:<+25.16e} Δ = {self.central_value-target:<+12.2e}')
            diff_perc = (self.central_value-target)/target*100
            if abs(diff_perc) < 1.:
                report[-1] += f' ({Colour.GREEN}{diff_perc:.3f}%{Colour.END}'
            else:
                report[-1] += f' ({Colour.RED}{diff_perc:.3f}%{Colour.END}'
            if abs(diff_perc/err_perc) < 3.:
                report[-1] += f' {Colour.GREEN} = {abs(diff_perc/err_perc):.2f}σ{Colour.END})'
            else:
                report[-1] += f' {Colour.RED} = {abs(diff_perc/err_perc):.2f}σ{Colour.END})'

        # Join all lines and return
        return '\n'.join(f'| > {line}' for line in report)

class Triangle(object):

    def __init__(self, m_psi: float, m_s: float, p: LorentzVector, q: LorentzVector):
        self.m_psi = m_psi
        self.m_s = m_s
        self.p = p
        self.q = q

        # Only perform sanity checks if in the physical region
        if (self.p+self.q).squared() > 0. or self.p.squared() > 0. or self.q.squared() > 0.:
            if m_s <= 0.:
                raise TrianglerException('m_s must be positive.')
            if abs(p.squared()) / m_s > TOLERANCE:
                raise TrianglerException('p must be on-shell.')
            if abs(q.squared()) / m_s > TOLERANCE:
                raise TrianglerException('q must be on-shell.')
            if abs((p+q).squared()-m_s**2)/m_s**2 > TOLERANCE:
                raise TrianglerException('p+q must be on-shell.')

    def parameterize(self, xs: list[float], parameterisation: str, origin: Vector | None = None) -> tuple[Vector, float]:
        match parameterisation:
            case 'cartesian': return self.cartesian_parameterize(xs, origin)
            case 'spherical': return self.spherical_parameterize(xs, origin)
            case _ : raise TrianglerException(f'Parameterisation {parameterisation} not implemented.')

    def cartesian_parameterize(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        def poly_map(x):
            return 1/(1-x)-1/x
        def poly_map_jac(x):
            return 1/(1-x)**2 + 1/x**2
        if origin is None:
            origin = Vector(0,0,0)
        return origin + Vector(*(poly_map(x) for x in xs)), math.prod(poly_map_jac(x) for x in xs)
        raise NotImplementedError("Implement cartesian parameterization in function 'cartesian_parameterize'. Ex. 2.5")

    def spherical_parameterize(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        r = xs[0]/(1-xs[0])
        phi = xs[2] * math.pi
        cos_th = 1-xs[1]*2
        sin_th = math.sqrt(1-cos_th**2)

        v = Vector(
            r * sin_th * math.cos(phi),
            r * sin_th * math.sin(phi),
            r * cos_th
        )
        if origin is None:
            origin = Vector(0,0,0)
        return origin + v, 2 * (2 * math.pi) * (r**2 / (1-xs[0])**2)
        raise NotImplementedError("Implement cartesian parameterization in function 'spherical_parameterize'. Ex. 2.5")

    def integrand_xspace(self, xs: list[float], parameterization: str, integrand_implementation: str, improved_ltd: bool = False, multi_channeling: bool | int = True) -> float:
        try:
            if multi_channeling is False:
                k, jac = self.parameterize(xs, parameterization)
                wgt = self.integrand(k, integrand_implementation, improved_ltd)
                final_wgt = wgt * jac
            else:
                final_wgt = 0.
                multi_channeling_power = 3
                if multi_channeling is True or multi_channeling == 0:
                    raise NotImplementedError('Implement multi-channeled integrand. Ex. 2.11')
                if multi_channeling is True or multi_channeling == 1:
                    raise NotImplementedError('Implement multi-channeled integrand. Ex. 2.11')
                if multi_channeling is True or multi_channeling == 2:
                    raise NotImplementedError('Implement multi-channeled integrand. Ex. 2.11')

            if math.isnan(final_wgt):
                logger.debug(f"Integrand evaluated to NaN at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. Setting it to zero")
                final_wgt = 0.
        except ZeroDivisionError:
            logger.debug(f"Integrand divided by zero at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. Setting it to zero")
            final_wgt = 0.

        return final_wgt

    def integrand(self, loop_momentum: Vector, integrand_implementation: str, improved_ltd: bool = False) -> float:
        try:
            match integrand_implementation:
                case 'python': return self.python_integrand(loop_momentum, improved_ltd)
                case 'rust' : return self.rust_integrand(loop_momentum, improved_ltd)
                case _ : raise TrianglerException(f'Integrand implementation {integrand_implementation} not implemented.')
        except ZeroDivisionError:
            logger.debug(f"Integrand divided by zero for k = [{Colour.BLUE}{', '.join(f'{ki:+.16e}' for ki in loop_momentum.to_list())}{Colour.END}]. Setting it to zero")
            return 0.

    def python_integrand(self, loop_momentum: Vector, improved_ltd: bool = False) -> float:

        q = [LorentzVector(0,0,0,0), LorentzVector(0,0,0,0)-self.q, self.p]

        E = [math.sqrt((loop_momentum + qi.spatial()).squared() + self.m_psi**2) for qi in q]
        def term(i, j, k):
            return 1/(2*E[i]*
                      (E[i]+E[j]+(q[i].t-q[j].t))*
                      (E[i]-E[j]+(q[i].t-q[j].t))*
                      (E[i]+E[k]+(q[i].t-q[k].t))*
                      (E[i]-E[k]+(q[i].t-q[k].t))
                      )
        if not improved_ltd:
            return (2 * math.pi)**-3*(term(0,1,2)+term(1,0,2)+term(2,0,1))
            raise NotImplementedError("Implement basic LTD expression in function 'python_integrand'. Ex. 2.2")

        else:
            raise NotImplementedError("OPTIONAL: implement improved LTD expression in function 'python_integrand'. Ex. 2.3")

    def rust_integrand(self, loop_momentum: Vector, improved_ltd: bool = False) -> float:
        if not improved_ltd:
            raise NotImplementedError('Rust integrand is only implemented for the improved LTD version.')
        else:
            raise NotImplementedError("OPTIONAL: implement improved LTD expression in rust in function 'rust_integrand' and file 'src/lib.rs'. Ex. 2.4")
            # Use the code below to call the rust integrand compiled in src/lib.rs
            return ltd_triangle(self.m_psi,
                         [loop_momentum.x, loop_momentum.y, loop_momentum.z],
                         [self.p.t, self.p.x, self.p.y, self.p.z],
                         [self.q.t, self.q.x, self.q.y, self.q.z])

    def integrate(self, integrator: str, parameterisation: str, integrand_implementation: str, improved_ltd: bool, target: float | None = None, **opts) -> IntegrationResult:

        match integrator:
            case 'naive': return self.naive_integrator(parameterisation, integrand_implementation, improved_ltd, target, **opts)
            case 'vegas': return Triangle.vegas_integrator(self, parameterisation, integrand_implementation, improved_ltd, target, **opts)
            case 'symbolica': return self.symbolica_integrator(parameterisation, integrand_implementation, improved_ltd, target, **opts)
            case _ : raise TrianglerException(f'Integrator {integrator} not implemented.')

    def naive_worker(self, n_points: int, call_args: list[Any]) -> IntegrationResult:
        res = IntegrationResult(0., 0.)
        start_time = time.time()
        for _ in range(n_points):
            xs = [random.random() for _ in range(3)]
            wgt = self.integrand_xspace(xs, *call_args)
            if res.max_wgt is None or abs(wgt) > abs(res.max_wgt):
                res.max_wgt = wgt
                res.max_wgt_point = xs
            res.central_value += wgt
            res.error += wgt**2
            res.n_samples += 1
        res.elapsed_time += time.time() - start_time
        return res

    def naive_integrator(self, parameterisation: str, integrand_implementation: str, improved_ltd: bool, target, **opts) -> IntegrationResult:

        integration_result = IntegrationResult(0., 0.)

        function_call_args = [parameterisation, integrand_implementation, improved_ltd, opts['multi_channeling']]
        for i_iter in range(opts['n_iterations']):
            logger.info(f'Naive integration: starting iteration {Colour.GREEN}{i_iter+1}/{opts["n_iterations"]}{Colour.END} using {Colour.BLUE}{opts["points_per_iteration"]}{Colour.END} points ...')
            if opts['n_cores'] > 1:
                raise NotImplementedError('Implement parallelisation in naive integrator. Ex. 2.9')
            else:
                integration_result.combine_with(self.naive_worker(opts['points_per_iteration'],function_call_args))
            # Normalize a copy for temporary printout
            processed_result = copy.deepcopy(integration_result)
            processed_result.normalize()
            logger.info(f'... result after this iteration:\n{processed_result.str_report(target)}')

        # Normalize results
        integration_result.normalize()

        return integration_result

    @staticmethod
    def vegas_worker(triangle: Triangle, id: int, all_xs: list[list[float]], call_args: list[Any]) -> tuple[int, list[float], IntegrationResult]:
        res = IntegrationResult(0., 0.)
        t_start = time.time()
        all_weights = []
        for xs in all_xs:
            weight = triangle.integrand_xspace(xs, *call_args)
            all_weights.append(weight)
            if res.max_wgt is None or abs(weight) > abs(res.max_wgt):
                res.max_wgt = weight
                res.max_wgt_point = xs
            res.central_value += weight
            res.error += weight**2
            res.n_samples += 1
        res.elapsed_time += time.time() - t_start

        return (id, all_weights, res)

    @staticmethod
    def vegas_functor(triangle: Triangle, res: IntegrationResult, n_cores: int, call_args: list[Any]) -> Callable[[list[list[float]]],list[float]]:
        
        @vegas.batchintegrand
        def f(all_xs):
            all_weights = []
            if n_cores > 1:
                all_args = [(copy.deepcopy(triangle), i_chunk, all_xs_split, call_args) for i_chunk, all_xs_split in enumerate(chunks(all_xs, len(all_xs)//n_cores+1))]
                with multiprocessing.Pool(processes=n_cores) as pool:
                    all_results = pool.starmap(Triangle.vegas_worker, all_args)
                for _id, wgts, this_result in sorted(all_results, key=lambda x: x[0]):
                    all_weights.extend(wgts)
                    res.combine_with(this_result)
                return all_weights
            else:
                _id, wgts, this_result = Triangle.vegas_worker(triangle, 0, all_xs, call_args)
                all_weights.extend(wgts)
                res.combine_with(this_result)
            return all_weights

        return f

    def vegas_integrator(self, parameterisation: str, integrand_implementation: str, improved_ltd: bool, _target, **opts) -> IntegrationResult:

        integration_result = IntegrationResult(0., 0.)

        integrator = vegas.Integrator(3 * [[0, 1],])

        local_worker = Triangle.vegas_functor(self, integration_result, opts['n_cores'], [parameterisation, integrand_implementation, improved_ltd, opts['multi_channeling']])
        # Adapt grid
        integrator(local_worker, nitn=opts['n_iterations'], neval=opts['points_per_iteration'], analyzer=vegas.reporter())
        # Final result
        result = integrator(local_worker, nitn=opts['n_iterations'], neval=opts['points_per_iteration'], analyzer=vegas.reporter())

        integration_result.central_value = result.mean
        integration_result.error = result.sdev
        return integration_result

    def symbolica_integrator(self, parameterisation: str, integrand_implementation: str, improved_ltd: bool, target, **opts) -> IntegrationResult:

        integration_result = IntegrationResult(0., 0.)

        if opts['multi_channeling']:
            integrator =  NumericalIntegrator.discrete([
                NumericalIntegrator.continuous(3),
                NumericalIntegrator.continuous(3),
                NumericalIntegrator.continuous(3)
            ])
        else:
            integrator = NumericalIntegrator.continuous(3)

        for i_iter in range(opts['n_iterations']):
            logger.info(f'Symbolica integration: starting iteration {Colour.GREEN}{i_iter+1}/{opts["n_iterations"]}{Colour.END} using {Colour.BLUE}{opts["points_per_iteration"]}{Colour.END} points ...')
            samples = integrator.sample(opts['points_per_iteration'])
            raise NotImplementedError("Implement Symbolica integrator in function 'symbolica_integrator' (Ex. 2.12)")
            integrator.add_training_samples(samples, res)

            # Learning rate is 1.5
            avg, err, _chi_sq = integrator.update(1.5)
            integration_result.central_value = avg
            integration_result.error = err
            logger.info(f'... result after this iteration:\n{integration_result.str_report(target)}')

        return integration_result

    def analytical_result(self) -> complex:
        import numpy as np
        if self.m_s > 2 * self.m_psi:
            logger.critical('Analytical result not implemented for m_s > 2 * m_psi. Analytical result set to 0.')
            return complex(0.,0.)
        else:
            return 1/(8*np.pi**2) * 1/self.m_s**2 * np.arcsin(self.m_s/(2*self.m_psi))**2
    def plot(self, **opts):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fixed_x = None
        for i_x in range(3):
            if i_x not in opts['xs']:
                fixed_x = i_x
                break
        if fixed_x is None:
            raise TrianglerException('At least one x must be fixed (0,1 or 2).')
        n_bins = opts['mesh_size']
        # Create a grid of x and y values within the range [0., 1.]
        # Apply small offset to avoid divisions by zero
        offset = 1e-6
        x = np.linspace(opts['range'][0]+offset, opts['range'][1]-offset, n_bins)
        y = np.linspace(opts['range'][0]+offset, opts['range'][1]-offset, n_bins)
        X, Y = np.meshgrid(x, y)

        # Calculate the values of f(x, y) for each point in the grid
        Z = np.zeros((n_bins, n_bins))
        # Calculate the values of f(x, y) for each point in the grid using nested loops
        xs = [0.,]*3
        xs[fixed_x] = opts['fixed_x']
        for i in range(n_bins):
            for j in range(n_bins):
                xs[opts['xs'][0]] = X[i,j]
                xs[opts['xs'][1]] = Y[i,j]
                if opts['x_space']:
                    Z[i, j] = self.integrand_xspace(xs, opts['parameterisation'], opts['integrand_implementation'], opts['improved_ltd'], opts['multi_channeling'])
                else:
                    Z[i, j] = self.integrand(Vector(xs[0],xs[1],xs[2]), opts['integrand_implementation'], opts['improved_ltd'])

        # Take the logarithm of the function values, handling cases where the value is 0
        with np.errstate(divide='ignore'):
            log_Z = np.log10(np.abs(Z))
            log_Z[log_Z == -np.inf] = 0  # Replace -inf with 0 for visualization

        if opts['x_space']:
            xs = ['x0', 'x1', 'x2']
        else:
            xs = ['kx', 'ky', 'kz']
        xs[fixed_x] = str(opts['fixed_x'])
        

        if not opts['3D']:
            # Create the heatmap using matplotlib
            plt.figure(figsize=(8, 6))
            plt.imshow(log_Z, origin='lower', extent=[opts['range'][0], opts['range'][1], opts['range'][0], opts['range'][1]], cmap='viridis')
            plt.colorbar(label=f"log10(I({','.join(xs)}))")
        else:
            # Create a 3D plot
            raise NotImplementedError("Implement 3D plot functionality in function 'plot'. Ex. 2.7")

        plt.xlabel(f"{xs[opts['xs'][0]]}")
        plt.ylabel(f"{xs[opts['xs'][1]]}")
        plt.title(f"log10(I({','.join(xs)}))")
        plt.show()

if __name__ == '__main__':

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='Triangler')

    # Add options common to all subcommands
    parser.add_argument('--verbosity','-v', type=str, choices=['debug', 'info', 'critical'], default='info', help='Set verbosity level')
    parser.add_argument('--parameterisation', '-param', type=str,
                        choices=['cartesian','spherical'],
                        default='spherical',
                        help='Parameterisation to employ.')
    parser.add_argument('--improved_ltd', action='store_true', default = False, 
                        help='Use improved LTD expression which does not suffer from numerical instabilities.')
    parser.add_argument('--integrand_implementation', '-ii', type=str, default='python', choices=['python', 'rust'], help='Integrand implementation selected. Default = %(default)s')
    parser.add_argument('--multi_channeling', '-mc', action='store_true', default = False, 
                        help='Consider a multi-channeled integrand.')

    parser.add_argument('--m_s', type=float,
                        default=0.01,
                        help='Mass of the decaying scalar. Default = %(default)s GeV')
    parser.add_argument('-p', type=float, nargs=4,
                        default=[0.005, 0.0, 0.0, 0.005],
                        help='Four-momentum of the first photon. Default = %(default)s GeV')
    parser.add_argument('-q', type=float, nargs=4,
                        default=[0.005, 0.0, 0.0, -0.005],
                        help='Four-momentum of the second photon. Default = %(default)s GeV')

    # Add subcommands and their options
    subparsers = parser.add_subparsers(title="commands", dest="command",help='Various commands available')

    # create the parser for the "inspect" command
    parser_inspect = subparsers.add_parser('inspect', help='Inspect evaluation of a sample point of the integration space.')
    parser_inspect.add_argument('--point','-p', type=float, nargs=3, help='Sample point to inspect')
    parser_inspect.add_argument('--x_space', action='store_true', default = False, 
                        help='Inspect a point given in x-space. Default = %(default)s')
    parser_inspect.add_argument('--full_integrand', action='store_true', default = False, 
                        help='Inspect the complete integrand, incl. multi-channeling. Default = %(default)s')

    # create the parser for the "integrate" command
    parser_integrate = subparsers.add_parser('integrate', help='Integrate the loop amplitude.')
    parser_integrate.add_argument('--n_iterations','-n', type=int, default=10, help='Number of iterations to perform. Default = %(default)s')
    parser_integrate.add_argument('--points_per_iteration','-ppi', type=int, default=100000, help='Number of points per iteration. Default = %(default)s')
    parser_integrate.add_argument('--integrator','-it', type=str, default='naive', choices=['naive', 'symbolica', 'vegas'], help='Integrator selected. Default = %(default)s')
    parser_integrate.add_argument('--n_cores', '-nc', type=int, default=1, help='Number of cores to run with. Default = %(default)s')
    parser_integrate.add_argument('--seed', '-s', type=int, default=None, help='Specify random seed. Default = %(default)s')

    # Create the parser for the "plot" command
    parser_plot = subparsers.add_parser('plot', help='Plot the integrand.')
    parser_plot.add_argument('--xs', type=int, nargs=2, default=None, help='Chosen 2-dimension projection of the integration space')
    parser_plot.add_argument('--fixed_x', type=float, default=0.75, help='Value of x kept fixed: default = %(default)s')
    parser_plot.add_argument('--range', '-r', type=float, nargs=2, default=[0., 1.], help='range to plot. default = %(default)s')
    parser_plot.add_argument('--x_space', action='store_true', default = False, 
                                            help='Plot integrand in x-space. Default = %(default)s')
    parser_plot.add_argument('--3D', '-3D', action='store_true', default = False, 
                                            help='Make a 3D plot. Default = %(default)s')
    parser_plot.add_argument('--mesh_size', '-ms', type=int, default=300, help='Number of bins in meshing: default = %(default)s')

    # create the parser for the "analytical_result" command
    parser_analytical = subparsers.add_parser('analytical_result', help='Compute the analytical result if  m_s < 2 * m_psi')
    # add argument m_psi to main parser
    parser.add_argument('--m_psi', type=float, default = 0.02)

    if 'analytical_result' not in subparsers.choices.keys():
        raise NotImplementedError("Implement the 'analytical_result' subcommand (Ex. 2.1)")

    args = parser.parse_args()

    if not hasattr(args, 'm_psi'):
        raise NotImplementedError("Implement option 'm_psi' for the main command (Ex. 2.1)")

    match args.verbosity:
        case 'debug': logger.setLevel(logging.DEBUG)
        case 'info': logger.setLevel(logging.INFO)
        case 'critical': logger.setLevel(logging.CRITICAL)

    q_vec = LorentzVector(args.q[0], args.q[1], args.q[2], args.q[3])
    p_vec = LorentzVector(args.p[0], args.p[1], args.p[2], args.p[3])
    triangle = Triangle(args.m_psi, args.m_s, q_vec, p_vec)

    match args.command:

        case 'analytical_result':
            res = triangle.analytical_result()
            logger.info(f'{Colour.GREEN}Analytical result:{Colour.END} {res.real:+.16e} {res.imag:+.16e}j GeV^{{-2}}')

        case 'inspect':
            if args.full_integrand:
                res = triangle.integrand_xspace(args.point, args.parameterisation, args.integrand_implementation, args.improved_ltd, args.multi_channeling)
                logger.info(f"Full integrand evaluated at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in args.point)}{Colour.END}] : {Colour.GREEN}{res:+.16e}{Colour.END}")
            else:
                if args.x_space:
                    k_to_inspect, jacobian = triangle.parameterize(args.point, args.parameterisation)
                else:
                    k_to_inspect, jacobian  = Vector(*args.point), 1.
                res = triangle.integrand(k_to_inspect, args.integrand_implementation, args.improved_ltd)
                report = f"Integrand evaluated at loop momentum k = [{Colour.BLUE}{', '.join(f'{ki:+.16e}' for ki in k_to_inspect.to_list())}{Colour.END}] : {Colour.GREEN}{res:+.16e}{Colour.END}"
                if args.x_space:
                    report += f' (excl. jacobian = {jacobian:+.16e})'
                logger.info(report)
        
        case 'integrate':
            if args.seed is not None:
                random.seed(args.seed)
                logger.info("Note that setting the random seed only ensure reproducible results with the naive integrator and a single core.")

            if args.n_cores > multiprocessing.cpu_count():
                raise TrianglerException(f'Number of cores requested ({args.n_cores}) is larger than number of available cores ({multiprocessing.cpu_count()})')

            target = triangle.analytical_result()
            t_start = time.time()
            res = triangle.integrate(
                target=target.real, **vars(args)
            )
            integration_time = time.time() - t_start
            tabs = '\t'*5
            new_line = '\n'
            logger.info('-'*80)
            logger.info(f"Integration with settings below completed in {Colour.GREEN}{integration_time:.2f}s{Colour.END}:{new_line}"
                        f"{new_line.join(f'| {Colour.BLUE}{k:<30s}{Colour.END}: {Colour.GREEN}{pformat(v)}{Colour.END}' for k, v in vars(args).items())}"
                        f"{new_line}| {new_line}{res.str_report(target.real)}")
            logger.info('-'*80)
        
        case 'plot':
            if args.xs is None:
                xs = [0,1]
            else:
                xs = args.xs
            raise NotImplementedError("Implement the call to the 'plot' subcommand (Ex. 2.6)")
        case _:
            raise TrianglerException(f'Command {args.command} not implemented.')
        
