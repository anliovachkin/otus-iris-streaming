import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.concat_ws
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

object MLStreaming extends App {

  val modelPath = "/home/anliovachkin/projects/study/otus/otus-iris-pipeline/model"
  val checkpointLocation = "/tmp/checkpoint"

  val spark = SparkSession.builder()
    .appName("ML streaming for Iris classification")
    .config("spark.master", "local")
    .getOrCreate()

  import spark.implicits._

  val model = PipelineModel.load(modelPath)

  val schema = StructType(
    StructField("sepal_length", DoubleType, false) ::
      StructField("sepal_width", DoubleType, false) ::
      StructField("petal_length", DoubleType, false) ::
      StructField("petal_width", DoubleType, false) :: Nil)

  val input = spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:29092")
    .option("subscribe", "iris")
    .load()
    .selectExpr("CAST(value as String)")
    .as[String]
    .map(_.split(","))
    .map(value => (value(0).toDouble, value(1).toDouble, value(2).toDouble, value(3).toDouble))
    .toDF("sepal_length", "sepal_width", "petal_length", "petal_width")
    .drop("_1")
    .drop("_2")
    .drop("_3")
    .drop("_4")

  private val prediction = model.transform(input)

  val query = prediction
    .select(concat_ws(",", $"sepal_length", $"sepal_width",
      $"petal_length", $"petal_width", $"predictedLabel")
      .as("value"))
    .writeStream
    .format("kafka")
    .option("checkpointLocation", checkpointLocation)
    .outputMode("append")
    .option("kafka.bootstrap.servers", "localhost:29092")
    .option("topic", "prediction")
    .start()

  query.awaitTermination()
}
