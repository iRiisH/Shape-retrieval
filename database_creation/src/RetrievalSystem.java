import processing.core.*;

import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.util.*;
import java.util.stream.*;

import java.io.*;

import java.nio.file.*;

import javax.swing.*;

import java.awt.image.*;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.FlowLayout;


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

    // viewer object
    Viewer viewer;

    // directions of views
    float[][] angles = null;

    // path to store raw_features
    static String raw_features_prefix = "raw_features/model_";
    static String raw_features_suffix = ".txt";

    // default constructor use optimal hyperparameters according to article
    public RetrievalSystem(Viewer viewer){
        this.num_kernels = 4;
        this.n_tile = 4;
        this.feature_size = 0.2f;
        this.size_vocabulary = 3;
        this.num_views = 7; // 22, 52, 102, 202
        this.num_row_sample = 8;
        this.num_col_sample = 8;
        this.num_sample = this.num_row_sample * this.num_col_sample;
        this.kmeans_train_size = (int)1E6;
        this.num_features = this.num_kernels * this.n_tile * this.n_tile;
    	float[] theta = {0.f, (float)(Math.PI/4.), (float)(2.*Math.PI/4.), (float)(3.*Math.PI/4.)};
        this.fc = new FeatureComputer(theta,
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
            Viewer viewer,
            float[] theta){
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
        this.fc = new FeatureComputer(
                theta,
                this.num_row_sample,this.num_col_sample,this.feature_size,this.n_tile);
        this.viewer = viewer;
        this.angles = this.getAngles();
    }

    // retrieve pre-computed evenly distributed views
    private float[][] getAngles(){
    	float[][] angles = ReadViews.cart2cam(ReadViews.read_views(this.num_views));
        return angles;
    }

    /*
        External Codes: with self-explained names
    */
    public static Mat BufferedImage2Mat(BufferedImage im) {
        byte[] pixels = ((DataBufferByte) im.getRaster().getDataBuffer()).getData();
        Mat image = new Mat(im.getHeight(), im.getWidth(), CvType.CV_8UC3);
        image.put(0, 0, pixels);
        return image;
    }
    public static BufferedImage Image2BufferedImage(Image im){
        BufferedImage bi = new BufferedImage(
                im.getWidth(null),im.getHeight(null),
                BufferedImage.TYPE_3BYTE_BGR);
         Graphics bg = bi.getGraphics();
         bg.drawImage(im, 0, 0, null);
         bg.dispose();
         return bi;
    }
    public static Image PImage2Image(PImage img){
        return img.getImage();
    }

    /*
        compute the views of model loaded in viewer
    */
    private Mat[] render(){
        Mat[] views = new Mat[this.num_views];
        for(int i=0;i<this.num_views;i++){
            this.viewer.setAngle(this.angles[i]);
            this.viewer.draw();
            views[i] = new Mat();
            Imgproc.cvtColor(
                    BufferedImage2Mat(
                            Image2BufferedImage(
                                    PImage2Image(this.viewer.get()))),
                    views[i],Imgproc.COLOR_RGB2GRAY);
            Core.bitwise_not(views[i],views[i]);

            // DEBUG
            //Highgui.imwrite("bin/views/views_"+String.valueOf(i)+".jpg",views[i]);
        }
        return views;
    }

    /*
        Check whether a feature vector is blank
    */
    private boolean isZeros(float[] checked){
        for(int i=0;i<checked.length;i++){
            if(checked[i]>0.f)return false;
        }
        return true;
    }

    /*
        To sample features from a set of features
    */
    private float[][] sampleFeatures(int num_mesh){
        int total_local_features = num_mesh * this.num_views * this.num_sample;
        int train_size = Math.min(this.kmeans_train_size,total_local_features);
        float[][] sampled_features = new float[train_size][this.num_features];
        ArrayList<Integer> ids = new ArrayList<Integer>(total_local_features);
        for(int i=0;i<total_local_features;i++){ids.add(i);}
        Collections.shuffle(ids);

        for(int i=0,pos=0;i<train_size;i++){
        	if(pos>=ids.size()) break;
            while(true){
            	if(pos>=ids.size()) break;
                int cur = ids.get(pos);
                int a = cur/this.num_views/this.num_sample;
                int b = cur%this.num_views/this.num_sample;
                int c = cur%this.num_sample;
                float[] local_feature = this.readRawFeatures(a,b,c);
                if(isZeros(local_feature))pos++;
                else{
                    sampled_features[i] = local_feature;
                    break;
                }
            }
            pos++;
        }
        return sampled_features;
    }

    /*
        Transform a list of features to a list of histograms
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
        Get TFIDF weights from histograms
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
        Due to too many raw features, they are at first stored in different files

        getRawFeatureFilename:
                mesh_id to filen that stores its features

        saveRawFeatures:
                save raw features to file

        readRawFeatures:
                read raw features of certain model
    */
    private String getRawFeatureFilename(int id){
        return raw_features_prefix+String.valueOf(id)+raw_features_suffix;
    }
    private void saveRawFeatures(int id,float[][][] features){
        PrintWriter writer = null;
        try{
            writer = new PrintWriter(this.getRawFeatureFilename(id), "UTF-8");
        } catch(Exception e){System.out.println(e);System.exit(0);}
        writer.println(String.valueOf(this.num_views)+" "
                +String.valueOf(this.num_sample)+" "
                +String.valueOf(this.num_features));
        for(int i=0;i<this.num_views;i++){
            for(int j=0;j<this.num_sample;j++){
                String f = "";
                for(int k=0;k<this.num_features;k++){
                    f += String.valueOf(features[i][j][k])+" ";
                }
                writer.println(f);
            }
        }
        writer.close();
    }
    private float[][][] readRawFeatures(int mesh_id){
        String filename = this.getRawFeatureFilename(mesh_id);
        float[][][] res = null;
        try{
            FileReader fileReader = new FileReader(filename);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String[] line = bufferedReader.readLine().split("\\s");
            int n_v = Integer.parseInt(line[0]);
            int n_s = Integer.parseInt(line[1]);
            int n_f = Integer.parseInt(line[2]);
            res = new float[n_v][n_s][n_f];
            for(int i=0;i<n_v;i++){
                for(int j=0;j<n_s;j++){
                    line = bufferedReader.readLine().split("\\s");
                    for(int k=0;k<n_f;k++){
                        res[i][j][k] = Float.parseFloat(line[k]);
                    }
                }
            }
            bufferedReader.close();
        } catch(Exception e){System.out.println(e);System.exit(0);}
       return res;
    }
    private float[] readRawFeatures(int i,int j,int k){
    	float[] res = null;
        try{
            Stream<String> lines = Files.lines(Paths.get(this.getRawFeatureFilename(i)));
            String[] line = lines.skip(j*this.num_sample+k+1).findFirst().get().split("\\s");
            res = new float[this.num_features];
            System.out.println('l');
            System.out.println(line.length);
            for(int l=0;l<this.num_features;l++){
                res[l] = Float.parseFloat(line[l]);
            }
            System.out.println('e');

        } catch(Exception e){System.out.println(e);System.exit(0);}
        return res;
    }

    /*
        fit function of retrieval system
    */
    public void fit(String[] files){
        /*
            1. compute raw features for each mesh and store them into files

            2. train K-means

            3. discretize raw features into words and get histograms
        */

        //1
        System.out.println("Fitting...");
        System.out.println("Computing features for each mesh for each view...");
        for(int mesh_id=0;mesh_id<files.length;mesh_id++){
            System.out.format("Total meshes: %4d, Current mesh id: %4d\n",files.length,mesh_id);
            this.viewer.loadModel(files[mesh_id]);
            Mat[] imgs = this.render();
            saveRawFeatures(mesh_id,fc.computeFeature(imgs));
        }

        //2
        System.out.println("Sampling features...");
        float[][] sampled_features = this.sampleFeatures(files.length);
        System.out.println("Training K-Means...");
        KMeans kmeans = new KMeans(this.size_vocabulary);
        kmeans.fit(sampled_features);
        System.out.println("KMeans get centroids:");
        kmeans.printCentroids();
        this.centroids = kmeans.centroids;

        //3
        System.out.println("Computing histograms for each mesh for each view...");
        this.histograms = new float[files.length][this.num_views][this.size_vocabulary];
        for(int i=0;i<files.length;i++){
            this.histograms[i] = this.getHistograms(this.readRawFeatures(i));
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
