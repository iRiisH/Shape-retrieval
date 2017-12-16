import java.util.*;
import Jcg.geometry.*;
import Jcg.polyhedron.*;
import processing.core.*;

public class SurfaceMesh {

	double scaleFactor=50; // scaling factor: useful for 3d rendering
	static double occludingOffset = 0.05;
	Viewer view;
	public Polyhedron_3<Point_3> polyhedron3D;
	private Point_3[] points;

	/*
		get polyhedron_3 from SharedVertexRepresentation:

		1. create new polyhedron_3 instance

		2. add vertices

		3. add edges and faces

	*/
	public Polyhedron_3<Point_3> sv2polyhedron(SharedVertexRepresentation sv){

		Polyhedron_3<Point_3> pl;
		pl = new Polyhedron_3<Point_3>(
				sv.sizeVertices,
				sv.sizeHalfedges,
				sv.sizeFaces);

		// add vertices
		Point_3[] ps = sv.points;
		this.points = ps;
		this.centralize();

		for(int i=0;i<ps.length;i++){
			pl.vertices.add(new Vertex<Point_3>(ps[i]));
		}

		// add edges and faces
		int n = sv.sizeVertices;
		HashMap<Integer,Halfedge<Point_3> > veMap; // to find opposite halfedge
		// use hashing: start_vertex_id * num_vertex + terminal_vertex_id
		veMap = new HashMap<Integer,Halfedge<Point_3> >();
		int[][] fs = sv.faces;
		for(int i=0;i<fs.length;i++){
			Face<Point_3> f = new Face<Point_3>(); // create new face
			pl.facets.add(f);
			ArrayList<Halfedge<Point_3> > hs; // to find next and prev halfedge
			hs = new ArrayList<Halfedge<Point_3> >();
			for(int j=0;j<fs[i].length;j++){
				Halfedge<Point_3> h = new Halfedge<Point_3>(); // create new halfedge
				pl.halfedges.add(h);
				int s = fs[i][j], t = fs[i][(j+1)%fs[i].length];
				f.setEdge(h);hs.add(h); veMap.put(n*s+t,h);
				h.face = f;
				if(veMap.containsKey(n*t+s)){
					Halfedge<Point_3> o = veMap.get(n*t+s);
					o.opposite = h; h.opposite = o;
				}
				h.vertex = pl.vertices.get(t);
			}
			for(int j=0;j<hs.size();j++){
				Halfedge<Point_3> prev = hs.get(j);
				Halfedge<Point_3> next = hs.get((j+1)%hs.size());
				prev.next = next; next.prev = prev;
			}
		}
		return pl;
	}

	/**
	 * Create a surface mesh from an OFF file
	 */
	public SurfaceMesh(Viewer view, String filename) {
		this.view=view;
		System.out.println(filename);
    	SharedVertexRepresentation sharedVertex=new SharedVertexRepresentation(filename);

		// this step is different from TD
		polyhedron3D = sv2polyhedron(sharedVertex);
		
    	this.points = new Point_3[this.polyhedron3D.vertices.size()];
    	int i = 0;
    	for (Vertex<Point_3> v: this.polyhedron3D.vertices)
    	{
    		this.points[i] = v.getPoint();
    		i++;
    	}
    	this.scaleFactor=this.computeScaleFactor();
	}

	/*
		centralize the model
	*/
	private void centralize(){
		float[] means = new float[3];
		float l = this.points.length;
		for(int i=0;i<this.points.length;i++){
			means[0] += (this.points[i].getX().floatValue()/l);
			means[1] += (this.points[i].getY().floatValue()/l);
			means[2] += (this.points[i].getZ().floatValue()/l);
		}
		for(int i=0;i<this.points.length;i++){
			this.points[i] = new Point_3(
					this.points[i].getX().floatValue()-means[0],
					this.points[i].getY().floatValue()-means[1],
					this.points[i].getZ().floatValue()-means[2]);
		}
	}

	public void drawAllTriangles(int v1,int v2,int v3,int v4){
		view.beginShape(view.TRIANGLES);
		for(Face<Point_3> f: this.polyhedron3D.facets) {
			Halfedge<Point_3> e=f.getEdge();
			ArrayList<Point_3> ps = new ArrayList<Point_3>();
			Halfedge<Point_3> h=e;
			while(true){
				ps.add(h.vertex.getPoint());
				h = h.next;
				if(h.equals(e))break;
			}

			view.noStroke();
			view.fill(v1,v2,v3,v4); // color of the triangle
			this.drawTriangle(ps); // draw a triangle face
		}
		view.endShape();
	}

	/*
		Contour
	*/
	public void geniusOcclidingCoutours(ArcBall.Vec3 pointOfView,float stroke_width){

		/*
			calculate the view direction
		*/
		PMatrix3D mat = (PMatrix3D)this.view.getMatrix(); mat.invert();
		float[] z_axis = {0.f,0.f,1.f,0.f};
		float[] cur_axis = new float[4]; mat.mult(z_axis,cur_axis);
		pointOfView = new ArcBall.Vec3(cur_axis[0],cur_axis[1],cur_axis[2]);

		/*
		 	We take all the edges with their two faces in
			different directions projected on view direction
		*/
		view.strokeWeight(stroke_width); // line width (for edges)
		view.stroke(20);
		for(Halfedge<Point_3> e: this.polyhedron3D.halfedges) {
			Halfedge<Point_3> en = e.next, o = e.opposite, on;

			/*
				If is boundary, draw
			*/
			if(o==null){
				while(!en.next.equals(e)){
					en = en.next;
				}
				this.drawSegment(e.vertex.getPoint(), en.vertex.getPoint());
				continue;
			}

			/*
				calculate normal vectors and judge whether to draw
			*/
			on = o.next;
			Point_3 u = e.getVertex().getPoint(),  v = o.getVertex().getPoint(),
			 		w = en.getVertex().getPoint(), t = on.getVertex().getPoint();
			float[][] coords = new float[4][3];
			coords[0][0] = u.getX().floatValue();
			coords[0][1] = u.getY().floatValue();
			coords[0][2] = u.getZ().floatValue();
			coords[1][0] = v.getX().floatValue();
			coords[1][1] = v.getY().floatValue();
			coords[1][2] = v.getZ().floatValue();
			coords[2][0] = w.getX().floatValue();
			coords[2][1] = w.getY().floatValue();
			coords[2][2] = w.getZ().floatValue();
			coords[3][0] = t.getX().floatValue();
			coords[3][1] = t.getY().floatValue();
			coords[3][2] = t.getZ().floatValue();
			ArcBall.Vec3 ae = new ArcBall.Vec3(
					coords[0][0]-coords[1][0],
					coords[0][1]-coords[1][1],
					coords[0][2]-coords[1][2]);
			ArcBall.Vec3 be = new ArcBall.Vec3(
					coords[2][0]-coords[0][0],
					coords[2][1]-coords[0][1],
					coords[2][2]-coords[0][2]);
			ArcBall.Vec3 ao = new ArcBall.Vec3(
					coords[1][0]-coords[0][0],
					coords[1][1]-coords[0][1],
					coords[1][2]-coords[0][2]);
			ArcBall.Vec3 bo = new ArcBall.Vec3(
					coords[3][0]-coords[1][0],
					coords[3][1]-coords[1][1],
					coords[3][2]-coords[1][2]);
			ArcBall.Vec3 ce = ArcBall.Vec3.cross(ae,be); ce.normalize();
			ArcBall.Vec3 co = ArcBall.Vec3.cross(ao,bo); co.normalize();
			float ze = ArcBall.Vec3.dot(pointOfView,ce);
			float zo = ArcBall.Vec3.dot(pointOfView,co);
			float eps = 1E-2f;
			if (false
				|| (ze*zo <= occludingOffset)) this.drawSegment(u, v);
		}
		view.strokeWeight(150);
		this.drawAllTriangles(255,255,255,255);
	}


	/*
		The following part is the code from TD
	*/
	/**
	 * Draw a segment between two points
	 */
	public void drawSegment(Point_3 p, Point_3 q) {
		float s=(float)this.scaleFactor;
		view.line(	(float)p.getX().doubleValue()*s, (float)p.getY().doubleValue()*s,
				(float)p.getZ().doubleValue()*s, (float)q.getX().doubleValue()*s,
				(float)q.getY().doubleValue()*s, (float)q.getZ().doubleValue()*s);
	}
	/**
	 * Draw a triangle face
	 */
	public void drawTriangle(ArrayList<Point_3> ps) {
		float s=(float)this.scaleFactor;
		for(int i=0;i<ps.size()-2;i++){
			Point_3 p = ps.get(i), q = ps.get(i+1), r = ps.get(i+2);
			view.vertex( (float)(p.getX().doubleValue()*s), (float)(p.getY().doubleValue()*s), (float)(p.getZ().doubleValue()*s));
			view.vertex( (float)(q.getX().doubleValue()*s), (float)(q.getY().doubleValue()*s), (float)(q.getZ().doubleValue()*s));
			view.vertex( (float)(r.getX().doubleValue()*s), (float)(r.getY().doubleValue()*s), (float)(r.getZ().doubleValue()*s));
		}
	}
	/**
	 * Draw the entire mesh
	 */
	public void draw() {
		//this.drawAxis();

		view.beginShape(view.TRIANGLES);
		for(Face<Point_3> f: this.polyhedron3D.facets) {
			Halfedge<Point_3> e=f.getEdge();
			ArrayList<Point_3> ps = new ArrayList<Point_3>();
			Halfedge<Point_3> h=e;
			while(true){
				ps.add(h.vertex.getPoint());
				h = h.next;
				if(h.equals(e))break;
			}
			// Point_3 p=e.vertex.getPoint();
			// Point_3 q=e.getNext().vertex.getPoint();
			// Point_3 r=e.getNext().getNext().vertex.getPoint();

			view.noStroke();
			view.fill(200,200,200,255); // color of the triangle
			this.drawTriangle(ps); // draw a triangle face
		}
		view.endShape();

		view.strokeWeight(2); // line width (for edges)
		view.stroke(20);
		for(Halfedge<Point_3> e: this.polyhedron3D.halfedges) {
			Point_3 p=e.vertex.getPoint();
			Halfedge<Point_3> h = e;
			while(!h.next.equals(e)){
				h = h.next;
			}
			Point_3 q=h.vertex.getPoint();

			this.drawSegment(p, q); // draw edge (p,q)
		}
		view.strokeWeight(1);
	}
	/**
	 * Draw the X, Y and Z axis
	 */
	public void drawAxis() {
		double s=1;
		Point_3 p000=new Point_3(0., 0., 0.);
		Point_3 p100=new Point_3(s, 0., 0.);
		Point_3 p010=new Point_3(0.,s, 0.);
		Point_3 p011=new Point_3(0., 0., s);

		drawSegment(p000, p100);
		drawSegment(p000, p010);
		drawSegment(p000, p011);
	}
	/**
	 * Return the value after truncation
	 */
	public static double round(double x, int precision) {
		return ((int)(x*precision)/(double)precision);
	}
	/**
	 * Compute the scale factor (depending on the max distance of the point set)
	 */
	public double computeScaleFactor() {
		if(this.polyhedron3D==null || this.polyhedron3D.vertices.size()<1)
			return 1;
		double maxDistance=0.;
		Point_3 origin=new Point_3(0., 0., 0.);
		for(Vertex<Point_3> v: this.polyhedron3D.vertices) {
			double distance=Math.sqrt(v.getPoint().squareDistance(origin).doubleValue());
			maxDistance=Math.max(maxDistance, distance);
		}
		return Math.sqrt(3)/maxDistance*300;
	}
	public Point_3 mean()
	{
		int n = this.polyhedron3D.vertices.size();
		Point_3[] points = new Point_3[n];
		int i = 0 ;
		for (Vertex<Point_3> v: this.polyhedron3D.vertices)
		{
			points[i] = v.getPoint();
			i++;
		}
		Number[] coefs = new Number[n];
		for (int j = 0 ; j < n ; j++)
			coefs[j] = this.scaleFactor/n;
		return Point_3.linearCombination(points, coefs);
	}
}
