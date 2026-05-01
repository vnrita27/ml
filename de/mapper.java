import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;

public class MapperClass extends Mapper<LongWritable, Text, Text, IntWritable> {
    public void map(LongWritable key, Text value, Context c) throws IOException, InterruptedException {
        String[] words = value.toString().split(" ");
        for (String w : words) {
            c.write(new Text(w), new IntWritable(1));
        }
    }
}