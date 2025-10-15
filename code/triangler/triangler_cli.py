import logging
import math
logging.basicConfig(level=logging.INFO)
import triangler
import argparse

if __name__ == '__main__':

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='Triangler')

    # Add options common to all subcommands
    parser.add_argument('--parameterization', '-param', type=str,
                        choices=['cartesian','spherical'],
                        default='spherical',
                        help='Parameterization to employ.')

    parser.add_argument('--m_psi', type=float,
                        default=0.02,
                        help='Mass of the internal fermion. Default = %(default)s GeV')
    parser.add_argument('-p', type=float, nargs=4,
                        default=[0.005, 0.0, 0.0, 0.005],
                        help='Four-momentum of the first photon. Default = %(default)s GeV')
    parser.add_argument('-q', type=float, nargs=4,
                        default=[0.005, 0.0, 0.0, -0.005],
                        help='Four-momentum of the second photon. Default = %(default)s GeV')
    subparsers = parser.add_subparsers(title="commands", dest="command",help='Various commands available')

    #INSPECT
    parser_inspect = subparsers.add_parser('inspect', help='Inspect evaluation of a sample point of the integration space.')
    parser_inspect.add_argument('--x_space', action='store_true', default = False,
                        help='Inspect a point given in x-space. Default = %(default)s')
    parser_inspect.add_argument('--point','-p', type=float, nargs=3, help='Sample point to inspect')

    #INTEGRATE
    parser_integrate = subparsers.add_parser('integrate', help='Integrate the loop amplitude.')


    args = parser.parse_args()

    q_vec =   triangler.LVec(args.q[0], args.q[1], args.q[2], args.q[3])
    p_vec =  -triangler.LVec(args.p[0], args.p[1], args.p[2], args.p[3])

    m_psi = args.m_psi

    logger = logging.getLogger('Triangler')
    logger.info('test')


    triangle = triangler.new_naive(q_vec, p_vec, m_psi, "vegas", "improved", args.parameterization, logger)

    match args.command:

        case 'analytical_result':
            print('Todo')
        case 'inspect':
            point = triangler.Vec3(*args.point)
            if args.x_space:
                res = triangle.evaluate_parameterized(point)
            else:
                res = triangle.evaluate(point)
            logger.info(f"Inspection result{res}")
        case 'integrate':
            res = triangle.integrate()
            logger.info(f"Integration result{res}")