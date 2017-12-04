import Jcg.geometry.Point_3;

class DistPoint implements Comparable<DistPoint>{
		final double dist;
		final Point_3 point;
		public DistPoint(double dist,Point_3 point){
			this.dist = dist;
			this.point = point;
		}
		
		public int compareTo(DistPoint dp){
			return Double.valueOf(this.dist).compareTo(dp.dist);
		}

}