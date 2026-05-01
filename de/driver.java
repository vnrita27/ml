import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Driver {
    public static void main(String[] args) throws Exception {
        Configuration c = new Configuration();
        Job j = Job.getInstance(c, "wordcount");

        j.setJarByClass(Driver.class);
        j.setMapperClass(MapperClass.class);
        j.setReducerClass(ReducerClass.class);

        j.setOutputKeyClass(Text.class);
        j.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(j, new Path(args[0]));
        FileOutputFormat.setOutputPath(j, new Path(args[1]));

        System.exit(j.waitForCompletion(true) ? 0 : 1);
    }
}