import java.util.*;

public class Argsort{

    private class ArgsortElement implements Comparable<ArgsortElement>{
        int id;
        float val;
        public ArgsortElement(int i,float e){
            this.id = i; this.val = e;
        }
        @Override
        public int compareTo(ArgsortElement others){
            if (this.val - others.val < 0.f) return -1;
            if (this.val - others.val == 0.f) return 0;
            return 1;
        }
    }

    private ArgsortElement[] originals;
    public float[] sorted_vals;
    public int[] sorted_ids;

    public Argsort(float[] arr){
        int l = arr.length;
        this.originals = new ArgsortElement[l];
        for(int i=0;i<l;i++){
            this.originals[i] = new ArgsortElement(i,arr[i]);
        }
        Arrays.sort(this.originals);
        this.sorted_vals = new float[l];
        this.sorted_ids = new int[l];
        for(int i=0;i<l;i++){
            this.sorted_ids[i] = this.originals[i].id;
            this.sorted_vals[i] = this.originals[i].val;
        }
    }
}
