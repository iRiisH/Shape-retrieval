import java.util.HashMap;

import Jcg.geometry.*;
import Jcg.polyhedron.*;
import processing.core.*;

public class SurfaceMesh {

	double scaleFactor=50; // scaling factor: useful for 3d rendering
	Viewer view;
	public Polyhedron_3<Point_3> polyhedron3D; // triangle mesh
	private Point_3[] points;
	/**
	 * Create a surface mesh from an OFF file
	 */
	public SurfaceMesh(Viewer view, String filename) {
		this.view=view;

		// shared vertex representation of the mesh
		System.out.println(filename);
    	SharedVertexRepresentation sharedVertex=new SharedVertexRepresentation(filename);
    	LoadMesh<Point_3> load3D=new LoadMesh<Point_3>();

    	polyhedron3D=load3D.createTriangleMesh(sharedVertex.points,sharedVertex.faceDegrees,
				sharedVertex.faces,sharedVertex.sizeHalfedges);

    	polyhedron3D.isValid(false);
    	this.points = new Point_3[this.polyhedron3D.vertices.size()];
    	int i = 0;
    	for (Vertex<Point_3> v: this.polyhedron3D.vertices)
    	{
    		this.points[i] = v.getPoint();
    		i++;
    	}
    	this.scaleFactor=this.computeScaleFactor();
	}

	/**
	 * Draw a segment between two points
	 */
	public void drawSegment(Point_3 p, Point_3 q) {
		float s=(float)this.scaleFactor;
		this.view.line(	(float)p.getX().doubleValue()*s, (float)p.getY().doubleValue()*s,
				(float)p.getZ().doubleValue()*s, (float)q.getX().doubleValue()*s,
				(float)q.getY().doubleValue()*s, (float)q.getZ().doubleValue()*s);
	}

	/**
	 * Draw a triangle face
	 */
	public void drawTriangle(Point_3 p, Point_3 q, Point_3 r) {
		float s=(float)this.scaleFactor;
		view.vertex( (float)(p.getX().doubleValue()*s), (float)(p.getY().doubleValue()*s), (float)(p.getZ().doubleValue()*s));
		view.vertex( (float)(q.getX().doubleValue()*s), (float)(q.getY().doubleValue()*s), (float)(q.getZ().doubleValue()*s));
		view.vertex( (float)(r.getX().doubleValue()*s), (float)(r.getY().doubleValue()*s), (float)(r.getZ().doubleValue()*s));
	}


	/**
	 * Draw the entire mesh
	 */
	public void draw() {
		//this.drawAxis();

		view.beginShape(view.TRIANGLES);
		for(Face<Point_3> f: this.polyhedron3D.facets) {
			Halfedge<Point_3> e=f.getEdge();
			Point_3 p=e.vertex.getPoint();
			Point_3 q=e.getNext().vertex.getPoint();
			Point_3 r=e.getNext().getNext().vertex.getPoint();

			view.noStroke();
			view.fill(200,200,200,255); // color of the triangle
			this.drawTriangle(p, q, r); // draw a triangle face
		}
		view.endShape();

		view.strokeWeight(2); // line width (for edges)
		view.stroke(20);
		for(Halfedge<Point_3> e: this.polyhedron3D.halfedges) {
			Point_3 p=e.vertex.getPoint();
			Point_3 q=e.opposite.vertex.getPoint();

			this.drawSegment(p, q); // draw edge (p,q)
		}
		view.strokeWeight(1);
	}
	public void drawNormals(HashMap<Vertex, ArcBall.Vec3> normals_map)
	{
		//this.drawSegment(v.getPoint(), v.getPoint() + normals_map(v)); // draw edge (p,q)
	}
	public void occludingContours(ArcBall.Vec3 pointOfView) {
		
		double epsilon = .5;
		double[][] normals = NormalEstimator.computeNormals(this.points, 10);
		HashMap<Vertex, ArcBall.Vec3> normals_map = new HashMap<Vertex, ArcBall.Vec3>();
		int cnt = 0;
		for(Vertex v: this.polyhedron3D.vertices)
		{
			ArcBall.Vec3 norm = new ArcBall.Vec3(
					(float)normals[cnt][0],
					(float)normals[cnt][1],
					(float)normals[cnt][2]);
			normals_map.put(v, norm);
			cnt++;
		}
		

		view.strokeWeight(2); // line width (for edges)
		view.stroke(20);
		for(Halfedge<Point_3> e: this.polyhedron3D.halfedges) {
			
			Vertex<Point_3> p=e.vertex;
			Vertex<Point_3> q=e.opposite.vertex;
			if (Math.abs(ArcBall.Vec3.dot(normals_map.get(p), pointOfView)) < epsilon 
					&& Math.abs(ArcBall.Vec3.dot(normals_map.get(q), pointOfView)) < epsilon)
				this.drawSegment(p.getPoint(), q.getPoint()); // draw edge (p,q)
		}
		view.strokeWeight(1);
		
		view.beginShape(view.TRIANGLES);
		for(Face<Point_3> f: this.polyhedron3D.facets) {
			Halfedge<Point_3> e=f.getEdge();
			Point_3 p=e.vertex.getPoint();
			Point_3 q=e.getNext().vertex.getPoint();
			Point_3 r=e.getNext().getNext().vertex.getPoint();

			view.noStroke();
			view.fill(255,255,255,255); // color of the triangle
			this.drawTriangle(p, q, r); // draw a triangle face
		}
		view.endShape();
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
		return Math.sqrt(3)/maxDistance*150;
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

