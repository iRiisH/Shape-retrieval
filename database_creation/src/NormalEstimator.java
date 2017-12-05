import java.util.ArrayList;
import java.util.Collections;
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
		r[0] = (double)p.getX().doubleValue();
		r[1] = (double)p.getY().doubleValue();
		r[2] = (double)p.getZ().doubleValue();
		return r;
	}

	private static Point_3[] getKNearestNeighbors(Point_3[] ps,Point_3 q, int k) {
		Point_3[] nearests = new Point_3[k];
		ArrayList<DistPoint> dpl = new ArrayList<DistPoint>();
		int n = ps.length;
		for(int i=0;i<n;i++){
			Point_3 curp = ps[i];
			dpl.add(new DistPoint((double)q.distanceFrom(curp).doubleValue(),curp));
		}
		Collections.sort(dpl);
		for(int i=0;i<k;i++){
			nearests[i] = dpl.get(i).point;
		}
		return nearests;
	}

	/**
	 * Return the closest points to q, at distance at most d <p>
	 * <p>
	 * Warning: naive method, based on a linear scan
	 *
	 * @param sqRad  square of the distance (sqRad=d*d)
	 * @param q  point query
	 */
	private static List<Point_3> getClosestPoints(Point_3[] ps,Point_3 q, double sqRad) {
		ArrayList<Point_3> nearests = new ArrayList<Point_3>();
		ArrayList<DistPoint> dpl = new ArrayList<DistPoint>();
		int n = ps.length;
		for(int i=0;i<n;i++){
			Point_3 curp = ps[i];
			dpl.add(new DistPoint((double)q.distanceFrom(curp).doubleValue(),curp));
		}
		Collections.sort(dpl);

		int i=0;
		while(true){
			if(i==dpl.size()||dpl.get(i).dist>sqRad)break;
			nearests.add(dpl.get(i).point);i++;
		}
		return nearests;
	}
	/**
	 * Compute the outliers in the point cloud
	 *
	 * @param points  input point cloud
	 * @param k  number of closest neighbor
	 */
	public static double[][] computeNormals(Point_3[] ps, int k) {
		int n = ps.length;
		double[][] normals = new double[n][3];
		for(int i=0;i<n;i++){
			Point_3[] nearests = getKNearestNeighbors(ps,ps[i],k+1);
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
	public static double[][] computeNormals(Point_3[] ps, double sqRad) {
		int n = ps.length;

		double[][] normals = new double[n][3];

		for(int i=0;i<n;i++){

			List<Point_3> _nearests = getClosestPoints(ps,ps[i],sqRad);
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
}
