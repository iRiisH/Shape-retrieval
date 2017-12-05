import processing.core.*;

/**
 * A simple 3d viewer for visualizing surface meshes
 *
 * @author Luca Castelli Aleardi (INF555, 2012)
 *
 */
public class Viewer extends PApplet {

	SurfaceMesh mesh;

	int renderType=0; // choice of type of rendering
	int renderModes=3; // number of rendering modes

	int simplificationMethod=0;
	int nMethods=3; // number of simplification methods proposed
	int model_id = 0;
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

		filename = this.get_filename(model_id);
	  	this.mesh=new SurfaceMesh(this, filename);
	}

	public void draw() {

		// set the background color
	  	background(255);

		// set light
	  	directionalLight(101, 204, 255, -1, 0, 0);
	  	directionalLight(51, 102, 126, 0, -1, 0);
	  	directionalLight(51, 102, 126, 0, 0, -1);
	  	directionalLight(102, 50, 126, 1, 0, 0);
	  	directionalLight(51, 50, 102, 0, 1, 0);
	  	directionalLight(51, 50, 102, 0, 0, 1);

		// set original position
		/*
			positive direction x: right (opposite to left)
			positive direction y: down (opposite to up)
			positive direction z: close (opposite to far)
		*/
	  	translate(width/2.f,height/2.f,-1*height/2.f);

		// set stroke style
	  	this.strokeWeight(1);
	  	stroke(150,150,150);

	  	this.mesh.draw();
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
		//PApplet pa=new Viewer();
		//pa.setSize(400, 400);
		PApplet.main(new String[] { "Viewer" });
	}

}
