import java.util.List;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import Jcg.geometry.Point_3;
import Jcg.geometry.Vector_3;

/**
 * A class providing methods for the estimation of vertex normal in a 3D point cloud
 *
 * @author Luca Castelli Aleardi (INF555, 2016)
 *
 */
public class NormalEstimator {

	private static double[] point_3ToArray(Point_3 p){
		double[] r = new double[3];
		r[0] = (double)p.getX();
		r[1] = (double)p.getY();
		r[2] = (double)p.getZ();
		return r;
	}

	/**
	 * Compute the outliers in the point cloud
	 *
	 * @param points  input point cloud
	 * @param k  number of closest neighbor
	 */
	public static double[][] computeNormals(PointSet points, int k) {
		Point_3[] ps = points.toArray();
		int n = ps.length;
		double[][] normals = new double[n][3];
		for(int i=0;i<n;i++){
			Point_3[] nearests = points.getKNearestNeighbors(ps[i],k+1);
			double[][] pn = new double[k+1][3];
			for(int j=0;j<=k;j++){
				pn[j] = point_3ToArray(nearests[j]);
			}
			double[][] cov = new double[3][3];
			for(int j=1;j<=k;j++){
				for(int r=0;r<3;r++){
					for(int c=0;c<3;c++){
						cov[r][c] += (pn[j][r]-pn[0][r]) * (pn[j][c]-pn[0][c]);
					}
				}
			}
			Matrix C = new Matrix(cov);
			EigenvalueDecomposition eig = new EigenvalueDecomposition(C);
			Matrix V = eig.getV();
			normals[i] = V.transpose().getArray()[0];
		}
		return normals;
	}

	/**
	 * Compute the normals for all points in the point cloud
	 *
	 * @param points  input point cloud
	 * @param sqRad  distance parameter (sqRad=d*d)
	 */
	public static double[][] computeNormals(PointSet points, double sqRad) {
		Point_3[] ps = points.toArray();
		int n = ps.length;

		double[][] normals = new double[n][3];

		for(int i=0;i<n;i++){

			List<Point_3> _nearests = points.getClosestPoints(ps[i],sqRad);
			int k = _nearests.size()-1;
			if(k==0){continue;}
			Point_3[] nearests = new Point_3[k+1];_nearests.toArray(nearests);

			double[][] pn = new double[k+1][3];
			for(int j=0;j<=k;j++){
				pn[j] = point_3ToArray(nearests[j]);
			}
			double[][] cov = new double[3][3];
			for(int j=1;j<=k;j++){
				for(int r=0;r<3;r++){
					for(int c=0;c<3;c++){
						cov[r][c] += (pn[j][r]-pn[0][r]) * (pn[j][c]-pn[0][c]);
					}
				}
			}
			Matrix C = new Matrix(cov);
			EigenvalueDecomposition eig = new EigenvalueDecomposition(C);

			Matrix V = eig.getV();
			normals[i] = V.transpose().getArray()[0];
		}
		return normals;
	}

	/**
	 * Given a point p and a distance d, <p>
	 * compute the matrix $C=\sum_{i}^{k} [(p_i-P)(p_i-P)^t]$<p>
	 * <p>
	 * where $k$ is the number of points ${p_i}$ at distance at most $d$ from point $p$
	 *
	 * @param points  input point cloud
	 * @param p  the query point (for which we want to compute the normal)
	 * @param sqRad  squared distance (sqRad=d*d)
	 */
	public static Matrix getCovarianceMatrix(PointSet points, Point_3 p, double sqRad) {
		throw new Error("To be completed (TD6)");
	}

	/**
	 * Return the distance parameter (a rough approximation of the average distance between neighboring points)
	 *
	 * @param points  input point cloud
	 */
	public static double estimateAverageDistance(PointSet points) {
		int n=(int)Math.sqrt(points.size());
		double maxDistance=points.getMaxDistanceFromOrigin();
		return maxDistance*4/n;
	}

}
