import Jcg.geometry.Point_3;
import processing.core.*;

public class Viewer extends PApplet {

	static int nModel = 1815;
	static String path = "../../data/benchmark/db/";
	static int nMode = 2;

	SurfaceMesh mesh;
	ArcBall arcball;
	float scaling = 1.f;
	int mode = 0;

	int model_id = 333;
	String filename;

	private String get_filename(){
		String folder = String.valueOf(this.model_id / 100);
		String filename = String.valueOf(this.model_id);
		return path+folder+"/m"+filename+"/m"+filename+".off";
	}

	// initialization
	public void setup() {

		// initialize window size
	  	size(800,600,P3D);
		ortho(-width/2, width/2, -height/2, height/2);
		// initialize Arcball
	  	ArcBall arcball = new ArcBall(this);
	  	this.arcball = arcball;
		this.loadModel();
	  	// this.mesh.scaleFactor = 500.;
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
	  	// this.mesh.occludingContours(direction);
		this.mesh.geniusOcclidingCoutours(direction);
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

		if(this.mode==0)
			drawContours(direction);
	  	else if(this.mode==1)
			drawNormal();
	  	//this.mesh.draw();
	}

	public void loadModel(){
	  	this.mesh=new SurfaceMesh(this, this.get_filename());
		this.mesh.scaleFactor *= this.scaling;
	}

	public void keyPressed(){
		  switch(key) {
			case('n'):this.model_id=(this.model_id+1)%nModel;loadModel();break;
			case('p'):this.model_id=(this.model_id+nModel-1)%nModel;loadModel();break;
			case('L'):this.scaling *= 1.1;this.mesh.scaleFactor *= 1.1;break;
			case('S'):this.scaling /= 1.1;this.mesh.scaleFactor /= 1.1;break;
			case('M'):this.mode = (this.mode+1)%nMode;break;
			case('O'):SurfaceMesh.occludingOffset *= 1.1;break;
			case('o'):SurfaceMesh.occludingOffset /= 1.1;break;
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
