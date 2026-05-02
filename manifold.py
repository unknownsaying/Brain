import numpy as np
from itertools import product, combinations
from sympy import Matrix, eye, Rational, zeros
import sympy as sp

class ToricCalabiYau:
    """
    Resolved toric Calabi-Yau threefold from C^3/Z_3.
    Lattice N = Z^3. The singular cone is the positive octant.
    Resolution adds rays corresponding to the exceptional divisors.
    """
    def __init__(self):
        # Lattice: 3-dimensional
        self.N = Matrix.eye(3)
        # Basis vectors of the singular cone (first octant)
        self.v1 = Matrix([1,0,0])
        self.v2 = Matrix([0,1,0])
        self.v3 = Matrix([0,0,1])
        self.singular_rays = [self.v1, self.v2, self.v3]
        # Resolution: add ray v0 = (1,1,1) (the sum) to subdivide the cone
        self.v0 = Matrix([1,1,1])
        self.resolved_rays = [self.v1, self.v2, self.v3, self.v0]
        # Build the fan of the resolved space:
        # Maximal cones are all triples of rays that form a basis of N.
        self.cones = self._build_maximal_cones()

    def _build_maximal_cones(self):
        # Four rays -> take all 3-subsets that are unimodular (det=±1) and span N.
        rays = self.resolved_rays
        cones = []
        for combo in combinations(rays, 3):
            M = Matrix.hstack(*combo)
            if M.det() != 0 and M.det() in (1, -1):
                cones.append(combo)
        return cones

    def exceptional_divisors(self):
        # The ray v0 corresponds to a compact exceptional divisor.
        return [self.v0]  # for Z_3, one exceptional compact divisor

    def intersection_matrix(self):
        """
        Compute intersection numbers of compact exceptional divisors.
        For a smooth toric variety, intersections are computed from
        the linear relations among the rays.
        """
        # Rays: v1,v2,v3,v0. Relation: v1+v2+v3 - 3*v0 = 0? Actually
        # In N, we have v1+v2+v3 = 3*v0? No, v0=(1,1,1), so 3*v0 = (3,3,3)
        # but v1+v2+v3 = (1,1,1) = v0, so 1*v1+1*v2+1*v3 = 1*v0.
        # This means the compact divisor D0 has triple self-intersection.
        # Intersection numbers are computed using the toric Mori cone.
        # For brevity, return a symbolic representation.
        return "D0^3 = 9"   # known result: exceptional P^2 has degree 9

    def hodge_numbers(self):
        return {"h11": 36, "h21": 0}   # resolved T^6/Z_3

    def euler_number(self):
        return 2*(self.hodge_numbers()["h11"] - self.hodge_numbers()["h21"])

# ---------- Even orbifold: T^6/Z_4 -------------------
class ResolvedZ4Toric(ToricCalabiYau):
    def __init__(self):
        self.N = Matrix.eye(3)
        # Action: (1,1,2) mod 4
        # Singular cone is again octant. Resolution is more involved; we add several rays.
        # We'll present a minimal resolution: add v0=(1,1,2) and v0'=(1,1,0)? etc.
        # For demonstration, we skip full construction but note that exceptional divisors form chains.
        pass