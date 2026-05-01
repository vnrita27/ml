import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;

public class ReducerClass extends Reducer<Text, IntWritable, Text, IntWritable> {
    public void reduce(Text key, Iterable<IntWritable> values, Context c) throws IOException, InterruptedException {
        int s = 0;
        for (IntWritable v : values) {
            s += v.get();
        }
        c.write(key, new IntWritable(s));
    }
}