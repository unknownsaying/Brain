import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

# --------------------------
#  Six‑torus lattice
# --------------------------
class SixTorus:
    def __init__(self, R=1.0):
        """Simple cubic lattice in R^6 with radius R."""
        self.R = R
        self.lattice_basis = 2*np.pi*R * np.eye(6)

    def point_in_fundamental(self, coords):
        """Bring 6D coordinates into [0, 2πR) box."""
        return np.mod(coords, 2*np.pi*self.R)

# --------------------------
#  Orbifold Group Actions
# --------------------------
class OrbifoldAction:
    def __init__(self, order, phases):
        """
        phases: list of three integers (k1,k2,k3) such that
        action = (exp(2πi k1/n), exp(2πi k2/n), exp(2πi k2/n))
        and sum(k_i) = 0 mod n (Calabi-Yau condition).
        """
        self.order = order
        self.phases = phases
        # Calculate the rotation matrix in real 6D
        self.matrix_6d = self._make_rotation_matrix()

    def _make_rotation_matrix(self):
        """Build a 6x6 block diagonal rotation matrix from the phases."""
        n = self.order
        blocks = []
        for k in self.phases:
            theta = 2*np.pi*k/n
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
            blocks.append(rot)
        return sp.block_diag(*blocks)

    def act_on_point(self, point):
        """Apply orbifold transformation to a 6‑vector (real)."""
        return self.matrix_6d @ point

    def fixed_points(self, torus):
        """Find points in the fundamental domain that are fixed under the action,
        i.e. g·x = x + lattice vector."""
        # For a torus, fixed points satisfy (1 - g)x ∈ Λ.
        # We solve over the real numbers modulo lattice.
        # Here we list known analytic solutions for the simple lattice.
        n = self.order
        R = torus.R
        # For the common orbifolds, fixed points are combinations of
        # coordinates that are fractions of the lattice.
        # We'll generate all combinations of (0, 2πR/n, 2*2πR/n, ...)
        # for each complex plane that satisfy the fixed condition.
        # For simplicity, return the list of invariant sublattices.
        # For Z3 with phases (1,1,1), the fixed tori are isolated points.
        # For Z4 with phases (1,1,2), fixed points form complex lines.
        # We'll generate all points where each zi is fixed up to lattice.
        fixed_list = []
        for z1 in np.linspace(0, 2*np.pi*R, n, endpoint=False):
            for z2 in np.linspace(0, 2*np.pi*R, n, endpoint=False):
                for z3 in np.linspace(0, 2*np.pi*R, n, endpoint=False):
                    pt = np.array([z1, 0, z2, 0, z3, 0])  # embed as real
                    pt[:] = torus.point_in_fundamental(pt)
                    # Check invariance
                    gpt = self.act_on_point(pt)
                    diff = gpt - pt
                    # if diff is lattice vector (mod 2πR), it's fixed
                    if np.allclose(diff % (2*np.pi*R), 0, atol=1e-9):
                        fixed_list.append(pt.copy())
        # Remove duplicates from different representations
        unique = []
        for pt in fixed_list:
            if not any(np.allclose(pt, u) for u in unique):
                unique.append(pt)
        return unique

# --------------------------
#  Build the "odd" and "even" orbifolds
# --------------------------
R = 1.0
torus = SixTorus(R)

# Odd orbifold: Z3 with phases (1,1,1) -> Calabi-Yau
odd_orbifold = OrbifoldAction(order=3, phases=[1,1,1])
odd_fixed = odd_orbifold.fixed_points(torus)
print(f"Odd orbifold (Z3) – Number of fixed points: {len(odd_fixed)}")
# Should be 27 isolated fixed points (3^3)

# Even orbifold: Z4 with phases (1,1,2) -> sum=4≡0 mod4, Calabi-Yau
even_orbifold = OrbifoldAction(order=4, phases=[1,1,2])
even_fixed = even_orbifold.fixed_points(torus)
print(f"Even orbifold (Z4) – Number of fixed tori/points: {len(even_fixed)}")
# This yields fixed 2‑tori (sets of lines) – many points in this simple enumeration.

# --------------------------
#  3D Projection for visualisation
# --------------------------
def project_to_3d(points_6d):
    """Take the first three real coordinates as a naive projection."""
    return np.array([p[[0,2,4]] for p in points_6d])

odd3d = project_to_3d(odd_fixed)
even3d = project_to_3d(even_fixed)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(odd3d[:,0], odd3d[:,1], odd3d[:,2], c='r', s=40)
ax1.set_title("Odd Orbifold (Z3) Fixed Points")
ax1.set_xlabel('x1'); ax1.set_ylabel('x2'); ax1.set_zlabel('x3')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(even3d[:,0], even3d[:,1], even3d[:,2], c='b', s=10)
ax2.set_title("Even Orbifold (Z4) Fixed Points (subset)")
ax2.set_xlabel('x1'); ax2.set_ylabel('x2'); ax2.set_zlabel('x3')
plt.show()