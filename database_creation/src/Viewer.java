import Jcg.geometry.Point_3;
import processing.core.*;
import org.opencv.core.*;
public class Viewer extends PApplet {

	static int nModel = 1815;
	static String path = "../../data/benchmark/db/";
	static int nMode = 2;
	static float stroke_width = 4.f;
	SurfaceMesh mesh;
	ArcBall arcball;
	float scaling = 1.f;
	int mode = 0;

	int model_id = 333;
	String filename;
	RetrievalSystem rs;
	float[] angle = null;

	public static String id_to_path(int id){
		String folder = String.valueOf(id / 100);
		String filename = String.valueOf(id);
		return path+folder+"/m"+filename+"/m"+filename+".off";
	}

	private String get_filename(){
		return id_to_path(this.model_id);
	}

	// initialization
	public void setup() {

		// initialize window size
	  	size(800,600,P3D);
		// ortho(-width/2, width/2, -height/2, height/2);
		// initialize Arcball

		ArcBall arcball = new ArcBall(this);
	  	this.arcball = arcball;

		this.rs = new RetrievalSystem(this);
		String[] files = new String[nModel];
		for(int i=0;i<nModel;i++){
			files[i] = id_to_path(i);
		}
		rs.fit(files);


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
		try{
			directionalLight(255, 255, 255, -1, 0, 0);
		  	directionalLight(255, 255, 255, 0, -1, 0);
		  	directionalLight(255, 255, 255, 0, 0, -1);
		  	directionalLight(255, 255, 255, 1, 0, 0);
		  	directionalLight(255, 255, 255, 0, 1, 0);
		  	directionalLight(255, 255, 255, 0, 0, 1);
		} catch(Exception e){}
	  	// this.mesh.occludingContours(direction);
		this.mesh.geniusOcclidingCoutours(direction,stroke_width);
	}

	public void setAngle(float[] angle){
		this.angle = angle;
	}
	public void desetAngle(){
		this.angle = null;
	}

	public void draw() {

		// set the background color
	  	background(255);

		if(this.angle!=null){
			this.resetMatrix();
		}


		// set original position
		/*
			positive direction x: right (opposite to left)
			positive direction y: down (opposite to up)
			positive direction z: close (opposite to far)
		*/
	  	translate(0.f,0.f,-2*height/1.f);
	  	ArcBall.Quat q = this.arcball.q_now;
	  	ArcBall.Vec3 direction = new ArcBall.Vec3(q.x, q.y, q.z);
		if(this.angle!=null){
			this.rotateX(angle[0]);
			this.rotateY(angle[1]);
			this.rotateZ(angle[2]);
		}

		// set stroke style
	  	this.strokeWeight(1);
	  	stroke(150,150,150);
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

	public void loadModel(String filename){
		this.mesh=new SurfaceMesh(this, filename);
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
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.out.println(System.getProperty("java.version"));
		PApplet pa=new Viewer();
		pa.setSize(400, 400);
		PApplet.main(new String[] { "Viewer" });

	}

}
