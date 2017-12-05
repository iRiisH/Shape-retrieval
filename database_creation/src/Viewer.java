import Jcg.geometry.Point_3;
import processing.core.*;

/**
 * A simple 3d viewer for visualizing surface meshes
 *
 * @author Luca Castelli Aleardi (INF555, 2012)
 *
 */
public class Viewer extends PApplet {

	SurfaceMesh mesh;
	ArcBall arcball;
	int renderType=0; // choice of type of rendering
	int renderModes=3; // number of rendering modes

	int simplificationMethod=0;
	int nMethods=3; // number of simplification methods proposed
	int model_id = 500;
	static String path = "../../data/benchmark/db/";
	String filename;

	private static String get_filename(int model_id){
		String folder = String.valueOf(model_id / 100);
		String filename = String.valueOf(model_id);
		return path+folder+"/m"+filename+"/m"+filename+".off";
	}

	// initialization
	public void setup() {

		// initialize window size
	  	size(800,600,P3D);

		// initialize Arcball
	  	ArcBall arcball = new ArcBall(this);
	  	this.arcball = arcball;
		filename = this.get_filename(model_id);
	  	this.mesh=new SurfaceMesh(this, filename);
	  	this.mesh.scaleFactor = 500.;
	}
	public void drawNormal()
	{
		directionalLight(101, 204, 255, -1, 0, 0);
	  	directionalLight(51, 102, 126, 0, -1, 0);
	  	directionalLight(51, 102, 126, 0, 0, -1);
	  	directionalLight(102, 50, 126, 1, 0, 0);
	  	directionalLight(51, 50, 102, 0, 1, 0);
	  	directionalLight(51, 50, 102, 0, 0, 1);
	  	this.mesh.draw();	
		
	}
	public void drawContours(ArcBall.Vec3 direction)
	{
		directionalLight(255, 255, 255, -1, 0, 0);
	  	directionalLight(255, 255, 255, 0, -1, 0);
	  	directionalLight(255, 255, 255, 0, 0, -1);
	  	directionalLight(255, 255, 255, 1, 0, 0);
	  	directionalLight(255, 255, 255, 0, 1, 0);
	  	directionalLight(255, 255, 255, 0, 0, 1);
	  	this.mesh.occludingContours(direction);
	}
	public void draw() {

		// set the background color
	  	background(255);

		// set original position
		/*
			positive direction x: right (opposite to left)
			positive direction y: down (opposite to up)
			positive direction z: close (opposite to far)
		*/
	  	translate(width/2.f,height/2.f,-1*height/2.f);
	  	Point_3 mean = this.mesh.mean();
	  	translate(-mean.x.floatValue(), -mean.y.floatValue(), -mean.z.floatValue()); // center model
		// set stroke style
	  	this.strokeWeight(1);
	  	stroke(150,150,150);
	  	
	  	ArcBall.Quat q = this.arcball.q_now;
	  	ArcBall.Vec3 direction = new ArcBall.Vec3(q.x, q.y, q.z);
	  	drawContours(direction);
	  	//this.mesh.draw();
	}

	public void keyPressed(){
		  switch(key) {
			case('r'):this.renderType=(this.renderType+1)%this.renderModes; break;
		  }
	}

	/**
	 * For running the PApplet as Java application
	 */
	public static void main(String args[]) {
		PApplet pa=new Viewer();
		pa.setSize(400, 400);
		PApplet.main(new String[] { "Viewer" });
	}

}
