
import math

import numpy as np


def deg2rad(num):
    return num / 180 * math.pi


def rotateLine(vec,  deg):
    rad = deg2rad(deg)
    return np.array([np.cos(rad) * vec[0] - np.sin(rad) * vec[1], np.sin(rad) * vec[0] + np.cos(rad) * vec[1]])


def rad2deg(rad):
	return rad * 180.0 / math.pi


class M52D:
    def __init__(self, rZero, T, psi, xLength, isLeft=True, subdivisions=3):
        # rZero, T and psi are defined in Scherer et al.
        # Intraglottal pressure profiles for a symmetric and oblique glottis with a divergence angle of 10 degrees
        self.rZero = rZero
        self.T = T
        self.psi = deg2rad(psi)
        self.xLength = xLength
        self.subdivisions = subdivisions

        self.rPsi = self.rZero / (1.0 - np.sin(self.psi / 2.0))
        self.rL = self.T / 2.0
        self.r40 = self.T / 2.0
        self.b = np.sqrt(2.0) * self.rPsi / np.sqrt(1.0 + np.sin(self.psi / 2.0))
        self.Q1 = (self.T - self.rPsi) * (1.0 / np.cos(self.psi / 2.0)) + (self.rPsi - self.rL) * np.tan(self.psi / 2.0)
        self.Q2 = self.rL * np.sin(self.psi / 2.0)
        self.Q3 = self.Q1 * np.cos(self.psi / 2.0)
        self.Q4 = self.rZero
        self.Q5 = self.rL * np.sin(50)

        self.isLeft = isLeft

        self.generate()

    def subdivideSemicircle(self, center, direction, angle, subdivisions):
        pos = list()
        
        for i in range(subdivisions):
            pos.append(center + rotateLine(direction, (angle / subdivisions) * i))
        
        return pos

    def subdivideLine(self, a, b,  subdivisions):
        pos = list()
        
        direction = b - a
        length = np.sqrt(direction[0] * direction[0] + direction[1] * direction[1])
        direction = direction / length

        for i in range(subdivisions):
            pos.append(a + direction * ((length / subdivisions) * i))

        return pos

    def generate(self):
        start = np.array([0, 0])
        startEnd = np.array([self.xLength - self.rPsi, 0])

        pos1 = self.subdivideLine(start, startEnd, self.subdivisions)

        startFirstSemicircle = startEnd
        firstSemicircleOrigin = startEnd + np.array([0.0, self.rPsi])
        endFirstSemiCircle = firstSemicircleOrigin + rotateLine(np.array([0.0, -self.rPsi]), 90 - rad2deg(self.psi) / 2)
        pos2 = self.subdivideSemicircle(firstSemicircleOrigin, np.array([0.0, -self.rPsi]), 90 - rad2deg(self.psi) / 2, self.subdivisions - 1)

        startQ1 = endFirstSemiCircle
        endQ1 = startQ1 + rotateLine(np.array([self.Q1, 0.0]), 90 - rad2deg(self.psi) / 2)
        pos3 = self.subdivideLine(startQ1, endQ1, self.subdivisions - 1)

        directionToSecondCircleOrigin = rotateLine(np.array([self.rL, 0.0]), 180 - rad2deg(self.psi) / 2)
        secondSemiCircleOrigin = endQ1 + directionToSecondCircleOrigin
        secondSemiCircleEnd = secondSemiCircleOrigin + rotateLine(-directionToSecondCircleOrigin, 50 + (rad2deg(self.psi) / 2))
        pos4 = self.subdivideSemicircle(secondSemiCircleOrigin, -directionToSecondCircleOrigin, 50 + (rad2deg(self.psi) / 2), self.subdivisions - 1)

        lengthLastLine = (secondSemiCircleEnd[0] - start[0]) / np.cos(deg2rad(40.0))
        end = secondSemiCircleEnd + rotateLine(np.array([lengthLastLine, 0]), 180 - 40)
        pos5 = self.subdivideLine(secondSemiCircleEnd, end, self.subdivisions)
        pos6 = self.subdivideLine(end, start, self.subdivisions)

        self.vertices = pos1 + pos2 + pos3 + pos4 + pos5 + pos6

        maxX = 0
        if self.isLeft:
            for vertex in self.vertices:
                maxX = maxX if vertex[0] < maxX else vertex[0]
            for i in range(len(self.vertices)):
                self.vertices[i][0] -= maxX
        else:
            for vertex in self.vertices:
                maxX = maxX if vertex[0] < maxX else vertex[0]
            for i in range(len(self.vertices)):
                self.vertices[i][0] = -self.vertices[i][0] + maxX

        for i in range(len(self.vertices)):
            self.vertices[i][1] = -self.vertices[i][1]

        return self.vertices

    def translate(self, vec):        
        for i in range(len(self.vertices)):
            self.vertices[i] += vec

    def getVertices(self):
        return self.vertices
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import BSpline

    def catmull_rom_to_b_spline(points, resolution=100):
        """
        Converts a Catmull-Rom spline to a B-spline approximation.
        
        Parameters:
            points (np.ndarray): Nx2 array of 2D points.
            resolution (int): Number of points per segment for evaluation.

        Returns:
            Tuple (curve_x, curve_y): Evaluated B-spline points.
        """
        P = np.array(points)
        n = len(P)

        all_spline_pts = []

        for i in range(1, n - 2):
            # Four control points for Catmull-Rom
            p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]

            # Catmull-Rom basis matrix -> B-spline equivalent
            # Equivalent B-spline control points for this segment
            c0 = (-p0 + 3*p1 - 3*p2 + p3) / 6
            c1 = (3*p0 - 6*p1 + 3*p2) / 6
            c2 = (-3*p0 + 3*p2) / 6
            c3 = (p0 + 4*p1 + p2) / 6

            coeffs_x = [c0[0], c1[0], c2[0], c3[0]]
            coeffs_y = [c0[1], c1[1], c2[1], c3[1]]

            t = np.linspace(0, 1, resolution)
            T = np.vstack([t**3, t**2, t, np.ones_like(t)]).T

            segment_x = T @ coeffs_x
            segment_y = T @ coeffs_y

            all_spline_pts.append(np.column_stack([segment_x, segment_y]))

        return np.vstack(all_spline_pts)

    def catmull_rom_to_bspline_geomdl(points, method='centripetal'):
        """
        Converts a Catmull-Rom spline to a B-spline curve usable with geomdl.

        Parameters:
            points (list of list): Control points [[x, y], ...]
            method (str): 'uniform' or 'centripetal'

        Returns:
            BSpline.Curve: geomdl B-spline curve object
        """
        from geomdl import utilities

        degree = 3
        dim = 2
        points = np.array(points)

        # Step 1: Extend endpoints (phantom points)
        P = points
        P0 = 2 * P[0] - P[1]  # Mirror the first point
        Pn = 2 * P[-1] - P[-2]  # Mirror the last point
        extended_points = np.vstack([P0, P, Pn])

        # Step 2: Parameterize the control points (u vector)
        if method == 'centripetal':
            def alpha(p, q): return np.sqrt(np.linalg.norm(p - q))
        elif method == 'uniform':
            def alpha(p, q): return 1.0
        else:
            raise ValueError("method must be 'uniform' or 'centripetal'")

        u = [0.0]
        for i in range(1, len(extended_points)):
            u.append(u[i - 1] + alpha(extended_points[i], extended_points[i - 1]))
        u = np.array(u)
        u /= u[-1]  # Normalize to [0, 1]

        # Step 3: Compute knot vector for B-spline
        n = len(P)  # number of real control points
        m = n + degree + 1
        kv = [0] * (degree + 1)

        for i in range(1, n - degree):
            j = i + degree
            ui = sum(u[i + 1:i + degree + 1]) / degree
            kv.append(ui)

        kv += [1] * (degree + 1)

        # Step 4: Create B-spline curve
        curve = BSpline.Curve()
        curve.degree = degree
        curve.ctrlpts = P.tolist()
        curve.knotvector = kv

        return curve

    import matplotlib.pyplot as plt
    import numpy as np
    from geomdl import BSpline, fitting, knotvector
    from geomdl.visualization import VisMPL
    from geomdl.visualization import VisMPL as vis
    m5_test = M52D(0.1, 0.3, 0.0, 2.8)
    m5_test.subdivisions = 3
    m5_test = m5_test.generate()

    # Example 2D points (can be any polyline)
    points = np.array(m5_test)
    points = np.delete(points, [4, 5, 7, 8], axis=0)

    #points = np.concatenate([points, points[:1]], axis=0)

    curve = BSpline.Curve()
    curve.degree = 2
    curve.ctrlpts = points.tolist()
    curve.knotvector = knotvector.generate(curve.degree, len(curve.ctrlpts)) #catmull_rom_to_bspline_geomdl(points, method='centripetal')
    curve.delta = 0.01  # evaluation resolution
    curve_points = np.array(curve.evalpts)
    curve.vis = vis.VisCurve2D()
    curve.render()


    curve_poins = np.stack(points, np.zeros([points.shape[0], 1]))

    '''
    # Example points
    points = np.array([
        [0, 0],
        [1, 2],
        [2, 0.5],
        [3, 3],
        [4, 2.5],
        [5, 5],
    ])'''
    for i in range(1, 100, 1):
        b_spline_points = catmull_rom_to_b_spline(points, i)

        # Plotting
        #plt.scatter(points[:, 0], points[:, 1], label='Control Points')
        plt.plot(b_spline_points[:, 0], b_spline_points[:, 1], 'b-', label='Converted B-spline')
        plt.scatter(b_spline_points[:, 0], b_spline_points[:, 1], label='Converted B-spline')
        plt.legend()
        plt.title('Catmull-Rom to B-spline')
        plt.axis('equal')
        plt.show()

    exit()
    '''
    # Convert to list of [x, y, 0.0] (geomdl expects 3D)
    points_3d = [[p[0], p[1], 0.0] for p in points]

    # Choose degree and number of control points
    degree = 2

    for num_ctrlpts in range(4, 20, 1):
            
        # Fit the B-spline curve
        curve = fitting.approximate_curve(points_3d, degree, ctrlpts_size=num_ctrlpts)

        # Evaluate curve points
        curve.delta = 0.01  # evaluation resolution
        curve_points = np.array(curve.evalpts)
        curve.vis = vis.VisCurve2D()
        curve.render()

        # Plot original points and fitted B-spline
        #plt.plot(points[:, 0], points[:, 1], 'ro-', label='Original Line')
        #plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label=f'B-spline (n_ctrl={num_ctrlpts})')
        #plt.legend()
        #plt.axis('equal')
        #plt.title('B-spline with Minimized Control Points')
        #plt.show()
    '''