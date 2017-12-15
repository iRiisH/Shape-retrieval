import org.opencv.core.*;
import processing.core.*;
import java.util.*;
import java.awt.image.*;
// import java.awt.image.BufferedImage;
import java.util.*;
// import java.awt.*;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.FlowLayout;
import javax.swing.*;

public class RetrievalSystem{

    // number of views per mesh
    int num_views;

    // when sampling subimages to compute local features
    // number of rows
    int num_row_sample;
    // number of cols
    int num_col_sample;
    // number of total samples
    int num_sample;

    // number of Gabor filters
    int num_kernels;

    // number of tiles as indicated in article
    int n_tile;

    // feature size as indicated in the article
    float feature_size;

    // fraction of features used to train Kmeans
    int kmeans_train_size;

    // dimension of a feature vector (that used for Kmeans)
    int num_features;

    // size of vocabulary (dimension of histograms)
    int size_vocabulary;

    // prospective trained histograms for each model for each view
    float[][][] histograms;

    // prospective trained Kmeans centroids to discretize feature vectors
    float[][] centroids;

    // feature computer class to compute feature vectors from views
    FeatureComputer fc;

    Viewer viewer;

    float[][] angles = null;

    public RetrievalSystem(Viewer viewer){
        float stroke_width = Viewer.stroke_width;
        float omega = .02f, lambda = .3f, omega0 = .13f;
        this.num_kernels = 4;
        this.n_tile = 4;
        this.feature_size = 0.2f;
        this.size_vocabulary = 2500;
        this.num_views = 100;
        this.num_row_sample = 32;
        this.num_col_sample = 32;
        this.num_sample = this.num_row_sample * this.num_col_sample;
        this.kmeans_train_size = (int)1E6;
        float prf = omega0;
        float fb = stroke_width * omega;
        float ab = fb / lambda;
        float[] prfs = {prf,prf,prf,prf};
        float[] fos = {0.f,(float)Math.PI*.25f,(float)Math.PI*.5f,(float)Math.PI*.75f};
        float[] fbs = {fb,fb,fb,fb};
        float[] abs = {ab,ab,ab,ab};
        this.fc = new FeatureComputer(
                prfs,fos,fbs,abs,
                this.num_row_sample,
                this.num_col_sample,
                this.feature_size,
                this.n_tile);
        this.viewer = viewer;
        this.angles = this.getAngles();
    }

    public RetrievalSystem(
            int num_view_per_img,
            int num_row_sample_per_view,
            int num_col_sample_per_view,
            int num_kernels,
            int n_tile,
            float feature_size,
            int kmeans_train_size,
            int size_vocabulary,
            Viewer viewer){
        this.num_views = num_view_per_img;
        this.num_row_sample = num_row_sample_per_view;
        this.num_col_sample = num_col_sample_per_view;
        this.num_sample = this.num_row_sample * this.num_col_sample;
        this.num_kernels = num_kernels;
        this.n_tile = n_tile;
        this.feature_size = feature_size;
        this.kmeans_train_size = kmeans_train_size;
        this.num_features = this.num_kernels * this.n_tile * this.n_tile;
        this.size_vocabulary = size_vocabulary;
        float[][] params = this.getInitalKernelParams();
        this.fc = new FeatureComputer(
                params[0],params[1],params[2],params[3],
                this.num_row_sample,this.num_col_sample,this.feature_size,this.n_tile);
        this.viewer = viewer;
        this.angles = this.getAngles();
    }

    private float[][] getAngles(){
        throw new Error("Non implemented");
    }

    /*
        External Codes
    */
    public Mat BufferedImage2Mat(BufferedImage im) {
        // Convert INT to BYTE
        //im = new BufferedImage(im.getWidth(), im.getHeight(),BufferedImage.TYPE_3BYTE_BGR);
        // Convert bufferedimage to byte array
        byte[] pixels = ((DataBufferByte) im.getRaster().getDataBuffer())
                .getData();

        // Create a Matrix the same size of image
        Mat image = new Mat(im.getHeight(), im.getWidth(), CvType.CV_8UC3);
        // Fill Matrix with image values
        image.put(0, 0, pixels);

        return image;
    }
    public static BufferedImage Image2BufferedImage(Image im){
        BufferedImage bi = new BufferedImage
        (im.getWidth(null),im.getHeight(null),BufferedImage.TYPE_INT_RGB);
         Graphics bg = bi.getGraphics();
         bg.drawImage(im, 0, 0, null);
         bg.dispose();
         return bi;
    }
    public static Image PImage2Image(PImage img){
        return img.getImage();
    }

    /*
        receive a SurfaceMesh and compute its views
    */
    private Mat[] render(){
        Mat[] views = new Mat[this.num_views];
        for(int i=0;i<this.num_views;i++){
            this.viewer.setAngle(this.angles[i]);
            this.viewer.draw();
            views[i] = BufferedImage2Mat(
                    Image2BufferedImage(
                            PImage2Image(this.viewer.get())));
        }
        return views;
    }


    /*
        get parameters for Gabor filters
    */
    private float[][] getInitalKernelParams(){
        throw new Error("non implemented");
    }

    /*
        check whether a feature vector is blank
    */
    private boolean isZeros(float[] checked){
        for(int i=0;i<checked.length;i++){
            if(checked[i]>0.f)return false;
        }
        return true;
    }

    /*
        sample (v.) features from a set of features
    */
    private float[][] sampleFeatures(float[][][][] local_features){
        int num_mesh = local_features.length;
        int total_local_features = num_mesh * this.num_views * this.num_sample;
        int train_size = this.kmeans_train_size;
        float[][] sampled_features = new float[train_size][this.num_features];
        ArrayList<Integer> ids = new ArrayList<Integer>(total_local_features);
        for(int i=0;i<total_local_features;i++){ids.add(i);}
        Collections.shuffle(ids);
        for(int i=0,pos=0;i<train_size;i++){
            if(pos>=ids.size()) throw new Error("Too much zero features!");
            while(true){
                int cur = ids.get(pos);
                int a = cur/this.num_views/this.num_sample;
                int b = cur%this.num_views/this.num_sample;
                int c = cur%this.num_sample;
                if(isZeros(local_features[a][b][c]))pos++;
                else{
                    sampled_features[i] = local_features[a][b][c];
                    break;
                }
            }
            pos++;
        }
        return sampled_features;
    }

    /*
        transform a list of features to a list of histograms
    */
    private float[][] getHistograms(float[][][] features){
        float[][] histograms = new float[features.length][this.size_vocabulary];
        for(int i=0;i<features.length;i++){
            for(int j=0;j<this.num_sample;j++){
                if(this.isZeros(features[i][j]))continue;
                int mnid = 0;
                float mndist = KMeans.norm2(this.centroids[0],features[i][j]);
                for(int k=1;k<this.size_vocabulary;k++){
                    float dist = KMeans.norm2(this.centroids[k],features[i][j]);
                    if(dist < mndist){
                        mndist = dist;
                        mnid = k;
                    }
                }
                histograms[i][mnid]++;
            }
        }
        return histograms;
    }

    /*
        transform histograms to TFIDF weights
    */
    private void TFIDF(float[][][] histograms){
        float num_documents = histograms.length * this.num_views;
        float[] freq_documents = new float[this.size_vocabulary];
        for(int i=0;i<histograms.length;i++){
            for(int j=0;j<this.num_views;j++){
                for(int k=0;k<this.size_vocabulary;k++){
                    if(histograms[i][j][k]>0.f){
                        freq_documents[k]++;
                    }
                }
            }
        }
        for(int i=0;i<histograms.length;i++){
            for(int j=0;j<this.num_views;j++){
                float num_words = 0.f;
                for(int k=0;k<this.size_vocabulary;k++){
                    num_words += histograms[i][j][k];
                }
                for(int k=0;k<this.size_vocabulary;k++){
                    histograms[i][j][k] *= Math.log(num_documents/freq_documents[k])/num_words;
                }
            }
        }
    }

    /*
        fit function of retrieval system
    */
    public void fit(String[] files){

        System.out.println("Fitting...");
        System.out.println("Computing features for each mesh for each view...");
        float[][][][] local_features;
        local_features = new float[files.length][this.num_views][this.num_sample][this.num_features];

        for(int mesh_id=0;mesh_id<files.length;mesh_id++){
            System.out.format("Total meshes: %4d, Current mesh id: %4d\n",files.length,mesh_id);
            this.viewer.loadModel(files[mesh_id]);
            Mat[] imgs = this.render();
            local_features[mesh_id] = fc.computeFeature(imgs);
        }

        System.out.println("Sampling features...");
        float[][] sampled_features = this.sampleFeatures(local_features);

        System.out.println("Training K-Means...");
        KMeans kmeans = new KMeans(this.size_vocabulary);
        kmeans.fit(sampled_features);
        System.out.println("KMeans get centroids:");
        kmeans.printCentroids();
        this.centroids = kmeans.centroids;

        System.out.println("Computing histograms for each mesh for each view...");
        this.histograms = new float[files.length][this.num_views][this.size_vocabulary];
        for(int i=0;i<files.length;i++){
            this.histograms[i] = this.getHistograms(local_features[i]);
        }
        this.TFIDF(this.histograms);
    }

    /*
        Handy functions to compute cosine similarity between a and b vectors
    */
    public float cosineSimilarity(float[] a,float[] b){
        float norm_a = KMeans.norm2(a);
        float norm_b = KMeans.norm2(b);
        float inner_product = 0.f;
        for(int i=0;i<a.length;i++){
            inner_product += a[i] * b[i];
        }
        return inner_product / norm_a / norm_b;
    }

    /*
        compute similarities for each queries for each 3D meshes
    */
    private float[][] computeSimilarities(float[][] histos){
        int num_meshes = this.histograms.length;
        float[][] sims = new float[histos.length][num_meshes];
        for(int i=0;i<histos.length;i++){
            for(int j=0;j<num_meshes;j++){
                float mxsim = this.cosineSimilarity(this.histograms[j][0],histos[i]);
                for(int k=1;k<this.num_views;k++){
                    mxsim = Math.max(mxsim,this.cosineSimilarity(this.histograms[j][k],histos[i]));
                }
                sims[i][j] = mxsim;
            }
        }
        return sims;
    }

    /*
        give predictions according to similarities
    */
    private int[][] similarities2Predictions(float[][] sims,int num_predictions){
        int[][] predictions = new int[sims.length][num_predictions];
        for(int i=0;i<sims.length;i++){
            int[] ranks = new Argsort(sims[i]).sorted_ids;
            for(int j=0;j<num_predictions;j++){
                predictions[i][j] = ranks[j];
            }
        }
        return predictions;
    }

    /*
        predict function of retrieval system
    */
    public int[][] predict(Mat[] sketches,int num_predictions){
        float[][][] features = this.fc.computeFeature(sketches);
        float[][] sketches_histo = this.getHistograms(features);
        float[][] similarities = this.computeSimilarities(sketches_histo);
        int[][] predictions = this.similarities2Predictions(similarities,num_predictions);
        return predictions;
    }
}
