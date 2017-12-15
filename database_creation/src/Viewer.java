import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import Jcg.geometry.Point_3;
import processing.core.*;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class Viewer extends PApplet {

	static int nModel = 1815;
	static String path = "../data/benchmark/db/";
	static int nMode = 2;
	static float stroke_width = 4.f;
	SurfaceMesh mesh;
	ArcBall arcball;
	float scaling = 1.f;
	int mode = 0;

	int model_id = 332;
	String filename;
	RetrievalSystem rs;

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
		ortho(-width/2, width/2, -height/2, height/2);
		// initialize Arcball

		this.rs = new RetrievalSystem(this);
		String[] files = new String[nModel];
		for(int i=0;i<nModel;i++){
			files[i] = id_to_path(i);
		}
		rs.fit(files);

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
		this.mesh.geniusOcclidingCoutours(direction,stroke_width);
	}
	
	Mat toMat(PImage image)
	// converts processing PImage to openCV Mat
	{
		int w = image.width;
		int h = image.height;
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat mat = new Mat(h, w, CvType.CV_8UC4);
		Mat res = new Mat(h, w, CvType.CV_8UC3);
		byte[] data8 = new byte[w*h*4];
		int[] data32 = new int[w*h];
		PApplet.arrayCopy(image.pixels, data32);
		
		ByteBuffer bBuf = ByteBuffer.allocate(w*h*4);
		IntBuffer iBuf = bBuf.asIntBuffer();
		iBuf.put(data32);
		bBuf.get(data8);
		mat.put(0, 0, data8);
		for (int i= 0 ; i < w ; i++)
		{
			for (int j = 0 ; j < h ; j++)
			{
				double[] pix = mat.get(j, i);
				double[] new_pix = {pix[3], pix[2], pix[1]};
				res.put(j, i, new_pix);
			}
		}
		return res;
	}

	
	PImage toPImage(Mat mat) 
	{
		// TODO: correct this
		int w = mat.width();
		int h = mat.height();
		  
		PImage image = createImage(w, h, PApplet.ARGB);
		byte[] data8 = new byte[w*h*4];
		int[] data32 = new int[w*h];
		mat.get(0, 0, data8);
		ByteBuffer.wrap(data8).asIntBuffer().get(data32);
		PApplet.arrayCopy(data32, image.pixels);
		return image;
	}
	
	public void setAngle(float[] angle){
		throw new Error("Non implemented");
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

	public void save()
	{
		PImage screen = get();
		Mat test = toMat(screen);
		Highgui.imwrite("test.png", test);
		//screen.save("test.png");
	}
	

	public void keyPressed(){
		  switch(key) {
			case('n'):this.model_id=(this.model_id+1)%nModel;loadModel();break;
			case('p'):this.model_id=(this.model_id+nModel-1)%nModel;loadModel();break;
			case('L'):this.scaling *= 1.1;this.mesh.scaleFactor *= 1.1;break;
			case('S'):this.scaling /= 1.1;this.mesh.scaleFactor /= 1.1;break;
			case('M'):this.mode = (this.mode+1)%nMode;break;
			case('O'):SurfaceMesh.occludingOffset *= 1.1;break;
			case('s'):this.save();break;
			case('o'):SurfaceMesh.occludingOffset /= 1.1;break;
		  }
	}
	
	

	/**
	 * For running the PApplet as Java application
	 */
	public static void main(String args[]) {
		PApplet pa=new Viewer();
		pa.setSize(400, 400);
		System.out.println("beginning");
		System.out.println("Working Directory = " +
	              System.getProperty("user.dir"));
		
		PApplet.main(new String[] { "Viewer" });

		PImage pi = pa.createImage(400, 400, RGB);
		pi.save("test.png");
		for (int i = 0 ; i < pi.width ; i++)
		{
			for(int j = 0 ; j< pi.height ; j++)
			{
				int c = pi.get(i, j);
				//System.out.print(c);
			}
		}
	}
}
