import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_fan(rays, cones):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    # draw rays
    for r in rays:
        ax.quiver(0,0,0, r[0], r[1], r[2], color='k', arrow_length_ratio=0.1)
    # draw cones as triangles (simplified)
    for cone in cones:
        pts = np.array(cone).T
        # form a triangle from origin to the endpoints of the cone's rays
        # (we draw the convex hull of the three rays + origin)
        # For simplicity, connect the three rays as a wireframe
        edges = [(0,1), (1,2), (2,0)]
        for i,j in edges:
            ax.plot([pts[0,i], pts[0,j]], [pts[1,i], pts[1,j]], [pts[2,i], pts[2,j]], 'b-')
    ax.set_title("Resolved T^6/Z_3 fan (3D projection)")
    plt.show()

# Example
toric_cy = ToricCalabiYau()
plot_fan(toric_cy.resolved_rays, toric_cy.cones)