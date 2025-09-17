# Scala for ML Feature Preprocessing - Learning Roadmap

## Phase 1: Scala Fundamentals (Week 1-2)

### Basic Syntax & Types
```scala
// Variables and basic types
val immutableValue = 42
var mutableVariable = "hello"
val features: Array[Double] = Array(1.0, 2.5, 3.1)

// Functions for data transformation
def normalize(value: Double, min: Double, max: Double): Double =
  (value - min) / (max - min)

def scaleFeature(features: Array[Double]): Array[Double] = {
  val min = features.min
  val max = features.max
  features.map(normalize(_, min, max))
}
```

### Pattern Matching (crucial for data cleaning)
```scala
def handleMissingValue(value: Option[Double]): Double = value match {
  case Some(v) => v
  case None => 0.0 // or mean imputation
}

def categorizeAge(age: Int): String = age match {
  case a if a < 18 => "young"
  case a if a < 65 => "adult"
  case _ => "senior"
}
```

## Phase 2: Collections & Data Structures (Week 2-3)

### Essential Collections for ML Data
```scala
// Lists for sequential data
val features = List(1.0, 2.0, 3.0, 4.0)
val labels = List("A", "B", "A", "C")

// Maps for categorical encoding
val categoryMap = Map("A" -> 0, "B" -> 1, "C" -> 2)
val encodedLabels = labels.map(categoryMap)

// Arrays for numerical computations
val dataset = Array(
  Array(1.0, 2.0, 3.0),  // row 1
  Array(4.0, 5.0, 6.0),  // row 2
  Array(7.0, 8.0, 9.0)   // row 3
)

// Case classes for structured data
case class DataPoint(features: Array[Double], label: String)
val samples = Array(
  DataPoint(Array(1.0, 2.0), "positive"),
  DataPoint(Array(3.0, 4.0), "negative")
)
```

### Data Manipulation Operations
```scala
// Filtering and transforming data
val cleanData = dataset.filter(row => !row.contains(Double.NaN))
val squaredFeatures = dataset.map(_.map(math.pow(_, 2)))

// Grouping for statistical operations
val dataByLabel = samples.groupBy(_.label)
val meanByClass = dataByLabel.mapValues(points =>
  points.map(_.features(0)).sum / points.length
)
```

## Phase 3: Functional Programming for Data Pipelines (Week 3-4)

### Higher-Order Functions for Feature Engineering
```scala
// Composable transformations
def applyTransformation[T](data: List[T], transform: T => T): List[T] =
  data.map(transform)

def pipeline[T](data: List[T], transforms: List[T => T]): List[T] =
  transforms.foldLeft(data)((acc, transform) => acc.map(transform))

// Example: feature scaling pipeline
val scalingPipeline = List(
  (x: Double) => x - 5.0,  // center
  (x: Double) => x / 2.0   // scale
)

val rawFeatures = List(1.0, 3.0, 5.0, 7.0, 9.0)
val processedFeatures = pipeline(rawFeatures, scalingPipeline)
```

### Monadic Operations for Data Processing
```scala
// Option for handling missing data
def safeDivide(a: Double, b: Double): Option[Double] =
  if (b != 0) Some(a / b) else None

def processRatio(data: List[(Double, Double)]): List[Option[Double]] =
  data.map { case (num, denom) => safeDivide(num, denom) }

// Either for error handling in preprocessing
def validateFeature(value: Double): Either[String, Double] =
  if (value.isNaN) Left("NaN value detected")
  else if (value < 0) Left("Negative value not allowed")
  else Right(value)
```

## Phase 4: Feature Preprocessing Techniques (Week 4-6)

### Numerical Feature Processing
```scala
import scala.math._

object FeatureProcessor {
  // Standardization (Z-score normalization)
  def standardize(features: Array[Double]): Array[Double] = {
    val mean = features.sum / features.length
    val variance = features.map(x => pow(x - mean, 2)).sum / features.length
    val stdDev = sqrt(variance)
    features.map(x => (x - mean) / stdDev)
  }

  // Min-Max scaling
  def minMaxScale(features: Array[Double], newMin: Double = 0.0, newMax: Double = 1.0): Array[Double] = {
    val min = features.min
    val max = features.max
    val range = max - min
    features.map(x => newMin + (x - min) * (newMax - newMin) / range)
  }

  // Robust scaling (using median and IQR)
  def robustScale(features: Array[Double]): Array[Double] = {
    val sorted = features.sorted
    val median = sorted(sorted.length / 2)
    val q1 = sorted(sorted.length / 4)
    val q3 = sorted(3 * sorted.length / 4)
    val iqr = q3 - q1
    features.map(x => (x - median) / iqr)
  }
}
```

### Categorical Feature Encoding
```scala
object CategoricalEncoder {
  // One-hot encoding
  def oneHotEncode(categories: List[String]): Map[String, Array[Int]] = {
    val uniqueCategories = categories.distinct
    categories.map { cat =>
      cat -> uniqueCategories.map(unique => if (unique == cat) 1 else 0).toArray
    }.toMap
  }

  // Label encoding
  def labelEncode(categories: List[String]): (List[Int], Map[String, Int]) = {
    val categoryToIndex = categories.distinct.zipWithIndex.toMap
    val encoded = categories.map(categoryToIndex)
    (encoded, categoryToIndex)
  }

  // Frequency encoding
  def frequencyEncode(categories: List[String]): Map[String, Double] = {
    val counts = categories.groupBy(identity).mapValues(_.length.toDouble)
    val total = categories.length.toDouble
    counts.mapValues(_ / total)
  }
}
```

### Missing Data Handling
```scala
object MissingDataHandler {
  // Mean imputation
  def meanImputation(data: Array[Option[Double]]): Array[Double] = {
    val validValues = data.flatten
    val mean = validValues.sum / validValues.length
    data.map(_.getOrElse(mean))
  }

  // Forward fill
  def forwardFill[T](data: Array[Option[T]]): Array[Option[T]] = {
    data.scanLeft(None: Option[T]) { (last, current) =>
      current.orElse(last)
    }.tail
  }

  // K-nearest neighbors imputation (simplified)
  def knnImputation(data: Array[Array[Option[Double]]], k: Int = 3): Array[Array[Double]] = {
    data.map { row =>
      row.zipWithIndex.map { case (value, index) =>
        value.getOrElse {
          // Find k nearest complete rows and average their values at this index
          val completeRows = data.filter(_.forall(_.isDefined))
          val distances = completeRows.map(completeRow =>
            euclideanDistance(row.flatten, completeRow.flatten)
          )
          val kNearest = completeRows.zip(distances)
            .sortBy(_._2)
            .take(k)
            .map(_._1)
          kNearest.map(_(index).get).sum / k
        }
      }
    }
  }

  private def euclideanDistance(a: Array[Double], b: Array[Double]): Double = {
    sqrt(a.zip(b).map { case (x, y) => pow(x - y, 2) }.sum)
  }
}
```

## Phase 5: ML Libraries Integration (Week 6-8)

### Apache Spark with Scala
```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline

// Initialize Spark
val spark = SparkSession.builder()
  .appName("FeaturePreprocessing")
  .master("local[*]")
  .getOrCreate()

import spark.implicits._

// Load and preprocess data
val rawData = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("dataset.csv")

// String indexer for categorical features
val categoryIndexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

// One-hot encoder
val oneHotEncoder = new OneHotEncoder()
  .setInputCol("categoryIndex")
  .setOutputCol("categoryVector")

// Vector assembler to combine features
val assembler = new VectorAssembler()
  .setInputCols(Array("feature1", "feature2", "categoryVector"))
  .setOutputCol("features")

// Standard scaler
val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")

// Create preprocessing pipeline
val pipeline = new Pipeline()
  .setStages(Array(categoryIndexer, oneHotEncoder, assembler, scaler))

val model = pipeline.fit(rawData)
val processedData = model.transform(rawData)
```

### Breeze for Numerical Computing
```scala
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object BreezeFeatureProcessing {
  // Matrix operations for feature scaling
  def standardizeMatrix(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val means = mean(matrix(::, *)).t
    val stds = stddev(matrix(::, *)).t
    (matrix(*, ::) - means) /:/ stds
  }

  // PCA for dimensionality reduction
  def pca(data: DenseMatrix[Double], numComponents: Int): (DenseMatrix[Double], DenseVector[Double]) = {
    val centeredData = data(*, ::) - mean(data(::, *)).t
    val covariance = (centeredData.t * centeredData) / (data.rows - 1).toDouble
    val svd.SVD(u, s, vt) = svd(covariance)
    val components = vt(0 until numComponents, ::).t
    val transformed = centeredData * components
    (transformed, s(0 until numComponents))
  }

  // Feature selection using correlation
  def selectByCorrelation(features: DenseMatrix[Double], target: DenseVector[Double], threshold: Double): Array[Int] = {
    val correlations = (0 until features.cols).map { i =>
      val feature = features(::, i)
      math.abs(corrcoeff(feature, target))
    }
    correlations.zipWithIndex.filter(_._1 > threshold).map(_._2).toArray
  }
}
```

## Phase 6: Practical Projects (Week 8-10)

### Project 1: Customer Churn Prediction Preprocessing
```scala
// Build a complete preprocessing pipeline
case class Customer(
  age: Int,
  income: Double,
  category: String,
  monthsActive: Option[Int],
  totalSpent: Double
)

object ChurnPreprocessor {
  def preprocessCustomers(customers: List[Customer]): Array[Array[Double]] = {
    // Handle missing values
    val cleanedCustomers = customers.map { customer =>
      customer.copy(monthsActive = customer.monthsActive.orElse(Some(12)))
    }

    // Extract features
    val ages = cleanedCustomers.map(_.age.toDouble).toArray
    val incomes = cleanedCustomers.map(_.income).toArray
    val months = cleanedCustomers.map(_.monthsActive.get.toDouble).toArray
    val spending = cleanedCustomers.map(_.totalSpent).toArray

    // Scale numerical features
    val scaledAges = FeatureProcessor.standardize(ages)
    val scaledIncomes = FeatureProcessor.standardize(incomes)
    val scaledMonths = FeatureProcessor.standardize(months)
    val scaledSpending = FeatureProcessor.standardize(spending)

    // Encode categorical features
    val categories = cleanedCustomers.map(_.category)
    val (encodedCategories, _) = CategoricalEncoder.labelEncode(categories)

    // Combine features
    scaledAges.zip(scaledIncomes).zip(scaledMonths).zip(scaledSpending).zip(encodedCategories)
      .map { case ((((age, income), months), spending), category) =>
        Array(age, income, months, spending, category.toDouble)
      }
  }
}
```

### Project 2: Text Feature Extraction
```scala
object TextPreprocessor {
  import scala.util.matching.Regex

  // Clean text data
  def cleanText(text: String): String = {
    text.toLowerCase
      .replaceAll("[^a-zA-Z0-9\\s]", "")
      .replaceAll("\\s+", " ")
      .trim
  }

  // Create bag of words features
  def bagOfWords(documents: List[String], vocabSize: Int = 1000): Array[Array[Double]] = {
    val cleanedDocs = documents.map(cleanText)
    val allWords = cleanedDocs.flatMap(_.split(" ")).filter(_.nonEmpty)
    val vocabulary = allWords.groupBy(identity)
      .mapValues(_.length)
      .toSeq
      .sortBy(-_._2)
      .take(vocabSize)
      .map(_._1)
      .zipWithIndex
      .toMap

    cleanedDocs.map { doc =>
      val words = doc.split(" ")
      vocabulary.keys.map { word =>
        words.count(_ == word).toDouble
      }.toArray
    }.toArray
  }

  // TF-IDF features
  def tfidf(documents: List[String]): Array[Array[Double]] = {
    val cleanedDocs = documents.map(cleanText)
    val allWords = cleanedDocs.flatMap(_.split(" ")).distinct

    cleanedDocs.map { doc =>
      val words = doc.split(" ")
      allWords.map { word =>
        val tf = words.count(_ == word).toDouble / words.length
        val df = cleanedDocs.count(_.contains(word)).toDouble
        val idf = math.log(documents.length / df)
        tf * idf
      }.toArray
    }.toArray
  }
}
```

## Learning Resources & Next Steps

### Recommended Study Schedule:
- **Week 1-2**: Scala basics, syntax, pattern matching
- **Week 3-4**: Collections, functional programming concepts
- **Week 5-6**: Feature preprocessing techniques implementation
- **Week 7-8**: Spark and Breeze integration
- **Week 9-10**: Complete projects and advanced topics

### Practice Exercises:
1. Implement cross-validation data splitting
2. Build feature selection algorithms (mutual information, chi-square)
3. Create time series feature engineering functions
4. Develop automated feature pipeline with error handling

### Advanced Topics to Explore:
- Akka for distributed preprocessing
- Cats library for advanced functional programming
- ScalaTest for testing your preprocessing pipelines
- Performance optimization with parallel collections

This roadmap provides a structured approach to learning Scala specifically for ML feature preprocessing, with practical code examples and projects that build upon each other. Focus on understanding functional programming concepts as they're crucial for writing clean, composable data transformation pipelines.