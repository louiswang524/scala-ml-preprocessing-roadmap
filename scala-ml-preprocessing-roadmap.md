# Scala for ML Feature Preprocessing - Learning Roadmap

## Phase 1: Scala Fundamentals (Week 1-2)

### Basic Syntax & Types
```scala
// Variables and basic types
val immutableValue = 42
var mutableVariable = "hello"
val features: Array[Double] = Array(1.0, 2.5, 3.1)

// Type aliases for domain modeling
type FeatureVector = Array[Double]
type Label = String
type Dataset = Array[(FeatureVector, Label)]

// Functions for data transformation
def normalize(value: Double, min: Double, max: Double): Double =
  (value - min) / (max - min)

def scaleFeature(features: Array[Double]): Array[Double] = {
  val min = features.min
  val max = features.max
  features.map(normalize(_, min, max))
}

// Generic feature transformation function
def transformFeatures[T, U](data: Array[T], transform: T => U): Array[U] =
  data.map(transform)

// Example of using type parameters for flexible data processing
def processDataset[T: Numeric](data: Array[T], processor: T => Double): Array[Double] = {
  import Numeric.Implicits._
  data.map(processor)
}

// Algebraic Data Types for feature engineering
sealed trait FeatureType
case object Numerical extends FeatureType
case object Categorical extends FeatureType
case object Ordinal extends FeatureType
case object Binary extends FeatureType

case class Feature(name: String, featureType: FeatureType, values: Array[Any])
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

// Advanced pattern matching for data validation
def validateNumericFeature(value: Any): Either[String, Double] = value match {
  case d: Double if d.isNaN => Left("NaN detected")
  case d: Double if d.isInfinite => Left("Infinite value detected")
  case d: Double => Right(d)
  case i: Int => Right(i.toDouble)
  case s: String if s.matches("-?\\d+(\\.\\d+)?") => Right(s.toDouble)
  case _ => Left(s"Cannot convert $value to numeric")
}

// Pattern matching for feature type detection
def detectFeatureType(values: Array[String]): FeatureType = {
  val uniqueValues = values.distinct
  uniqueValues.length match {
    case 2 if uniqueValues.forall(v => v == "0" || v == "1" || v.toLowerCase == "true" || v.toLowerCase == "false") => Binary
    case n if n <= 10 && values.length > 50 => Categorical
    case _ if values.forall(_.matches("-?\\d+(\\.\\d+)?")) => Numerical
    case _ => Categorical
  }
}

// Pattern matching for outlier detection strategies
def detectOutliers(strategy: String, values: Array[Double]): Array[Boolean] = strategy match {
  case "iqr" => {
    val sorted = values.sorted
    val q1 = sorted(sorted.length / 4)
    val q3 = sorted(3 * sorted.length / 4)
    val iqr = q3 - q1
    val lowerBound = q1 - 1.5 * iqr
    val upperBound = q3 + 1.5 * iqr
    values.map(v => v < lowerBound || v > upperBound)
  }
  case "zscore" => {
    val mean = values.sum / values.length
    val stdDev = math.sqrt(values.map(v => math.pow(v - mean, 2)).sum / values.length)
    values.map(v => math.abs((v - mean) / stdDev) > 3)
  }
  case "isolation" => {
    // Simplified isolation forest logic
    val threshold = values.sum / values.length + 2 * math.sqrt(values.map(v => math.pow(v - values.sum / values.length, 2)).sum / values.length)
    values.map(_ > threshold)
  }
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

// Advanced data manipulation with for-comprehensions
def cleanAndTransformData(rawData: List[Map[String, Any]]): List[Map[String, Double]] = {
  for {
    record <- rawData
    if record.nonEmpty
    cleanedRecord = record.collect {
      case (key, value: Number) => key -> value.doubleValue()
      case (key, value: String) if value.matches("-?\\d+(\\.\\d+)?") => key -> value.toDouble
    }
    if cleanedRecord.nonEmpty
  } yield cleanedRecord
}

// Windowing operations for time series data
def createWindows[T](data: List[T], windowSize: Int, stepSize: Int = 1): List[List[T]] = {
  data.sliding(windowSize, stepSize).toList
}

// Statistical aggregations with parallel collections
def computeStatistics(data: Array[Double]): Map[String, Double] = {
  val parallelData = data.par
  Map(
    "mean" -> parallelData.sum / parallelData.length,
    "variance" -> parallelData.map(x => math.pow(x - parallelData.sum / parallelData.length, 2)).sum / parallelData.length,
    "min" -> parallelData.min,
    "max" -> parallelData.max,
    "median" -> {
      val sorted = parallelData.toArray.sorted
      if (sorted.length % 2 == 0) (sorted(sorted.length/2) + sorted(sorted.length/2-1)) / 2.0
      else sorted(sorted.length/2)
    }
  )
}

// Advanced grouping with custom aggregations
def groupAndAggregate[K, V](data: List[(K, V)],
                           aggregations: Map[String, List[V] => Double]): Map[K, Map[String, Double]] = {
  data.groupBy(_._1).mapValues { pairs =>
    val values = pairs.map(_._2)
    aggregations.mapValues(_(values))
  }
}
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

// Advanced pipeline with error handling
def safePipeline[T](data: List[T], transforms: List[T => Either[String, T]]): Either[String, List[T]] = {
  transforms.foldLeft(Right(data): Either[String, List[T]]) { (acc, transform) =>
    acc.flatMap(_.traverse(transform))
  }
}

// Implicit class for enhanced collection operations
implicit class EnhancedList[T](list: List[T]) {
  def mapWithIndex[U](f: (T, Int) => U): List[U] = list.zipWithIndex.map { case (item, idx) => f(item, idx) }
  def partitionMap[U, V](f: T => Either[U, V]): (List[U], List[V]) = {
    val (lefts, rights) = list.map(f).partition(_.isLeft)
    (lefts.map(_.left.get), rights.map(_.right.get))
  }
}

// Feature transformation with context preservation
case class FeatureWithContext[T](value: T, originalIndex: Int, metadata: Map[String, Any])

def preserveContext[T, U](data: List[T], transform: T => U): List[FeatureWithContext[U]] = {
  data.zipWithIndex.map { case (value, idx) =>
    FeatureWithContext(transform(value), idx, Map("timestamp" -> System.currentTimeMillis()))
  }
}

// Curried functions for reusable transformations
def createScaler(method: String): Double => Double => Double = method match {
  case "standardize" => mean => value => (value - mean) / 1.0 // simplified
  case "normalize" => max => value => value / max
  case _ => _ => identity
}

// Function composition for complex pipelines
def compose[A, B, C](f: B => C, g: A => B): A => C = a => f(g(a))
def andThen[A, B, C](f: A => B, g: B => C): A => C = a => g(f(a))

// Example usage
val normalizer = createScaler("normalize")(100.0)
val standardizer = createScaler("standardize")(50.0)
val combinedTransform = compose(standardizer, normalizer)
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

### Advanced Statistical Methods
```scala
import scala.util.Random

object StatisticalMethods {
  // Bootstrap sampling for robust statistics
  def bootstrap[T](data: Array[T], numSamples: Int, sampleSize: Int): Array[Array[T]] = {
    val random = new Random(42)
    (1 to numSamples).map { _ =>
      Array.fill(sampleSize)(data(random.nextInt(data.length)))
    }.toArray
  }

  // Stratified sampling for balanced datasets
  def stratifiedSample[T](data: Array[(T, String)], sampleRatio: Double): Array[(T, String)] = {
    val groupedData = data.groupBy(_._2)
    groupedData.flatMap { case (label, samples) =>
      val sampleSize = (samples.length * sampleRatio).toInt
      Random.shuffle(samples.toList).take(sampleSize)
    }.toArray
  }

  // Cross-validation fold creation
  def createKFolds[T](data: Array[T], k: Int): Array[Array[T]] = {
    val shuffled = Random.shuffle(data.toList)
    val foldSize = shuffled.length / k
    shuffled.grouped(foldSize).toArray
  }

  // Correlation matrix computation
  def correlationMatrix(features: Array[Array[Double]]): Array[Array[Double]] = {
    val numFeatures = features(0).length
    Array.tabulate(numFeatures, numFeatures) { (i, j) =>
      if (i == j) 1.0
      else {
        val featureI = features.map(_(i))
        val featureJ = features.map(_(j))
        pearsonCorrelation(featureI, featureJ)
      }
    }
  }

  private def pearsonCorrelation(x: Array[Double], y: Array[Double]): Double = {
    val n = x.length
    val meanX = x.sum / n
    val meanY = y.sum / n
    val numerator = x.zip(y).map { case (xi, yi) => (xi - meanX) * (yi - meanY) }.sum
    val denomX = math.sqrt(x.map(xi => math.pow(xi - meanX, 2)).sum)
    val denomY = math.sqrt(y.map(yi => math.pow(yi - meanY, 2)).sum)
    numerator / (denomX * denomY)
  }
}
```

### Numerical Feature Processing
```scala
import scala.math._
import scala.util.{Try, Success, Failure}

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

  // Quantile transformation
  def quantileTransform(features: Array[Double], nQuantiles: Int = 100): Array[Double] = {
    val sorted = features.sorted
    val quantiles = (0 until nQuantiles).map(i => sorted((i * sorted.length / nQuantiles).min(sorted.length - 1)))
    features.map { value =>
      val rank = quantiles.indexWhere(_ >= value)
      if (rank == -1) 1.0 else rank.toDouble / nQuantiles
    }
  }

  // Power transformation (Box-Cox)
  def boxCoxTransform(features: Array[Double], lambda: Double): Try[Array[Double]] = Try {
    if (features.exists(_ <= 0)) throw new IllegalArgumentException("Box-Cox requires positive values")
    if (lambda == 0) features.map(math.log)
    else features.map(x => (math.pow(x, lambda) - 1) / lambda)
  }

  // Yeo-Johnson transformation (handles negative values)
  def yeoJohnsonTransform(features: Array[Double], lambda: Double): Array[Double] = {
    features.map { x =>
      if (x >= 0) {
        if (lambda == 0) math.log(x + 1)
        else (math.pow(x + 1, lambda) - 1) / lambda
      } else {
        if (lambda == 2) -math.log(-x + 1)
        else -(math.pow(-x + 1, 2 - lambda) - 1) / (2 - lambda)
      }
    }
  }

  // Polynomial features generation
  def polynomialFeatures(features: Array[Double], degree: Int): Array[Double] = {
    (1 to degree).flatMap { d =>
      features.combinations(d).map(_.product)
    }.toArray
  }

  // Feature binning/discretization
  def equalWidthBinning(features: Array[Double], numBins: Int): Array[Int] = {
    val min = features.min
    val max = features.max
    val binWidth = (max - min) / numBins
    features.map { value =>
      val bin = ((value - min) / binWidth).toInt
      bin.min(numBins - 1)
    }
  }

  def equalFrequencyBinning(features: Array[Double], numBins: Int): Array[Int] = {
    val sorted = features.sorted
    val binSize = sorted.length / numBins
    val thresholds = (1 until numBins).map(i => sorted(i * binSize))
    features.map { value =>
      thresholds.indexWhere(_ > value) match {
        case -1 => numBins - 1
        case idx => idx
      }
    }
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

  // Target encoding (mean encoding)
  def targetEncode(categories: List[String], targets: List[Double]): Map[String, Double] = {
    require(categories.length == targets.length, "Categories and targets must have same length")
    categories.zip(targets).groupBy(_._1).mapValues { pairs =>
      val targetValues = pairs.map(_._2)
      targetValues.sum / targetValues.length
    }
  }

  // Binary encoding
  def binaryEncode(categories: List[String]): Map[String, Array[Int]] = {
    val uniqueCategories = categories.distinct
    val numBits = math.ceil(math.log(uniqueCategories.length) / math.log(2)).toInt
    val categoryToIndex = uniqueCategories.zipWithIndex.toMap

    categories.map { cat =>
      val index = categoryToIndex(cat)
      val binary = (0 until numBits).map(bit => (index >> bit) & 1).toArray
      cat -> binary
    }.toMap
  }

  // Hashing encoding for high cardinality features
  def hashingEncode(categories: List[String], numBuckets: Int): Map[String, Int] = {
    categories.map(cat => cat -> (cat.hashCode.abs % numBuckets)).toMap
  }

  // Ordinal encoding for ordered categories
  def ordinalEncode(categories: List[String], ordering: List[String]): Map[String, Int] = {
    val orderMap = ordering.zipWithIndex.toMap
    categories.map(cat => cat -> orderMap.getOrElse(cat, -1)).toMap
  }

  // Leave-one-out encoding (regularized target encoding)
  def leaveOneOutEncode(categories: List[String], targets: List[Double], alpha: Double = 1.0): Map[String, Double] = {
    require(categories.length == targets.length, "Categories and targets must have same length")
    val globalMean = targets.sum / targets.length

    categories.zip(targets).zipWithIndex.map { case ((cat, target), idx) =>
      val otherTargets = categories.zip(targets).zipWithIndex.collect {
        case ((c, t), i) if c == cat && i != idx => t
      }
      val localMean = if (otherTargets.nonEmpty) otherTargets.sum / otherTargets.length else globalMean
      val smoothed = (localMean * otherTargets.length + globalMean * alpha) / (otherTargets.length + alpha)
      cat -> smoothed
    }.toMap
  }
}
```

### Missing Data Handling
```scala
import scala.util.Random

object MissingDataHandler {
  // Mean imputation
  def meanImputation(data: Array[Option[Double]]): Array[Double] = {
    val validValues = data.flatten
    val mean = validValues.sum / validValues.length
    data.map(_.getOrElse(mean))
  }

  // Median imputation (more robust to outliers)
  def medianImputation(data: Array[Option[Double]]): Array[Double] = {
    val validValues = data.flatten.sorted
    val median = if (validValues.length % 2 == 0) {
      (validValues(validValues.length/2) + validValues(validValues.length/2-1)) / 2.0
    } else {
      validValues(validValues.length/2)
    }
    data.map(_.getOrElse(median))
  }

  // Mode imputation for categorical data
  def modeImputation[T](data: Array[Option[T]]): Array[T] = {
    val validValues = data.flatten
    val mode = validValues.groupBy(identity).maxBy(_._2.length)._1
    data.map(_.getOrElse(mode))
  }

  // Forward fill
  def forwardFill[T](data: Array[Option[T]]): Array[Option[T]] = {
    data.scanLeft(None: Option[T]) { (last, current) =>
      current.orElse(last)
    }.tail
  }

  // Backward fill
  def backwardFill[T](data: Array[Option[T]]): Array[Option[T]] = {
    data.reverse.scanLeft(None: Option[T]) { (last, current) =>
      current.orElse(last)
    }.tail.reverse
  }

  // Interpolation for time series data
  def linearInterpolation(data: Array[Option[Double]]): Array[Double] = {
    val result = data.clone()
    for (i <- data.indices) {
      if (data(i).isEmpty) {
        val prevIndex = (i - 1 to 0 by -1).find(j => data(j).isDefined).getOrElse(0)
        val nextIndex = (i + 1 until data.length).find(j => data(j).isDefined).getOrElse(data.length - 1)

        if (data(prevIndex).isDefined && data(nextIndex).isDefined) {
          val prevValue = data(prevIndex).get
          val nextValue = data(nextIndex).get
          val interpolated = prevValue + (nextValue - prevValue) * (i - prevIndex).toDouble / (nextIndex - prevIndex)
          result(i) = Some(interpolated)
        }
      }
    }
    result.map(_.getOrElse(0.0))
  }

  // Multiple imputation
  def multipleImputation(data: Array[Option[Double]], numImputations: Int = 5): Array[Array[Double]] = {
    val validValues = data.flatten
    (1 to numImputations).map { _ =>
      data.map(_.getOrElse(validValues(Random.nextInt(validValues.length))))
    }.toArray
  }

  // K-nearest neighbors imputation (enhanced)
  def knnImputation(data: Array[Array[Option[Double]]], k: Int = 3, weightFunc: Double => Double = d => 1.0 / (d + 1e-8)): Array[Array[Double]] = {
    data.map { row =>
      row.zipWithIndex.map { case (value, index) =>
        value.getOrElse {
          val completeRows = data.filter(_.forall(_.isDefined))
          if (completeRows.nonEmpty) {
            val distances = completeRows.map(completeRow =>
              euclideanDistance(row.flatten, completeRow.flatten)
            )
            val kNearest = completeRows.zip(distances)
              .sortBy(_._2)
              .take(k)

            // Weighted average based on distance
            val weights = kNearest.map(pair => weightFunc(pair._2))
            val weightedSum = kNearest.zip(weights).map { case ((nearestRow, _), weight) =>
              nearestRow(index).get * weight
            }.sum
            val totalWeight = weights.sum
            weightedSum / totalWeight
          } else {
            0.0 // fallback if no complete rows
          }
        }
      }
    }
  }

  // MICE (Multiple Imputation by Chained Equations) - simplified
  def miceImputation(data: Array[Array[Option[Double]]], maxIterations: Int = 10): Array[Array[Double]] = {
    var currentData = data.map(_.map(_.getOrElse(0.0))) // initial fill with zeros

    for (_ <- 1 to maxIterations) {
      for (colIndex <- data(0).indices) {
        val columnHasMissing = data.exists(_(colIndex).isEmpty)
        if (columnHasMissing) {
          // Simple linear regression imputation for this column
          val completeRows = data.zipWithIndex.filter { case (row, _) => row(colIndex).isDefined }
          if (completeRows.nonEmpty) {
            val targetValues = completeRows.map(_._1(colIndex).get)
            val predictorMeans = (0 until data(0).length).filter(_ != colIndex).map { predCol =>
              completeRows.map(pair => currentData(pair._2)(predCol)).sum / completeRows.length
            }

            // Impute missing values using mean (simplified regression)
            for ((row, rowIndex) <- data.zipWithIndex) {
              if (row(colIndex).isEmpty) {
                currentData(rowIndex)(colIndex) = targetValues.sum / targetValues.length
              }
            }
          }
        }
      }
    }
    currentData
  }

  private def euclideanDistance(a: Array[Double], b: Array[Double]): Double = {
    math.sqrt(a.zip(b).map { case (x, y) => math.pow(x - y, 2) }.sum)
  }

  // Missing data pattern analysis
  def analyzeMissingPatterns(data: Array[Array[Option[Double]]]): Map[String, Int] = {
    val patterns = data.map(row => row.map(_.isDefined).mkString(","))
    patterns.groupBy(identity).mapValues(_.length)
  }

  // Little's MCAR test (simplified)
  def testMCAR(data: Array[Array[Option[Double]]]): Boolean = {
    val missingMatrix = data.map(_.map(_.isEmpty))
    val totalMissing = missingMatrix.flatten.count(identity)
    val expectedMissing = data.length * data(0).length * 0.1 // assume 10% missing is expected
    math.abs(totalMissing - expectedMissing) < expectedMissing * 0.5
  }
}
```

## Phase 4.5: Advanced Feature Engineering Techniques

### Time Series Feature Engineering
```scala
import java.time.{LocalDateTime, Duration}

object TimeSeriesFeatures {
  case class TimeSeriesPoint(timestamp: LocalDateTime, value: Double, metadata: Map[String, Any] = Map())

  // Lag features
  def createLagFeatures(data: List[TimeSeriesPoint], lags: List[Int]): List[Map[String, Double]] = {
    data.zipWithIndex.map { case (point, index) =>
      val lagFeatures = lags.flatMap { lag =>
        if (index >= lag) Some(s"lag_$lag" -> data(index - lag).value)
        else None
      }.toMap
      lagFeatures + ("current" -> point.value)
    }
  }

  // Rolling window statistics
  def rollingStatistics(data: List[TimeSeriesPoint], windowSize: Int): List[Map[String, Double]] = {
    data.zipWithIndex.map { case (point, index) =>
      val windowStart = math.max(0, index - windowSize + 1)
      val window = data.slice(windowStart, index + 1).map(_.value)

      Map(
        "rolling_mean" -> (window.sum / window.length),
        "rolling_std" -> math.sqrt(window.map(v => math.pow(v - window.sum / window.length, 2)).sum / window.length),
        "rolling_min" -> window.min,
        "rolling_max" -> window.max,
        "rolling_median" -> {
          val sorted = window.sorted
          if (sorted.length % 2 == 0) (sorted(sorted.length/2) + sorted(sorted.length/2-1)) / 2.0
          else sorted(sorted.length/2)
        }
      )
    }
  }

  // Seasonal decomposition features
  def seasonalFeatures(data: List[TimeSeriesPoint], seasonLength: Int): List[Map[String, Double]] = {
    data.zipWithIndex.map { case (point, index) =>
      val seasonalIndex = index % seasonLength
      val sameSeasonValues = data.zipWithIndex.collect {
        case (p, i) if i % seasonLength == seasonalIndex && i != index => p.value
      }

      Map(
        "seasonal_mean" -> (if (sameSeasonValues.nonEmpty) sameSeasonValues.sum / sameSeasonValues.length else point.value),
        "seasonal_deviation" -> (point.value - (if (sameSeasonValues.nonEmpty) sameSeasonValues.sum / sameSeasonValues.length else point.value)),
        "season_index" -> seasonalIndex.toDouble
      )
    }
  }

  // Fourier transform features
  def fourierFeatures(data: List[Double], numComponents: Int): Array[Double] = {
    val n = data.length
    (1 to numComponents).flatMap { k =>
      val cosCoeff = data.zipWithIndex.map { case (value, i) =>
        value * math.cos(2 * math.Pi * k * i / n)
      }.sum * 2 / n

      val sinCoeff = data.zipWithIndex.map { case (value, i) =>
        value * math.sin(2 * math.Pi * k * i / n)
      }.sum * 2 / n

      List(cosCoeff, sinCoeff)
    }.toArray
  }
}
```

### Text Feature Engineering
```scala
object AdvancedTextPreprocessor {
  import scala.util.matching.Regex
  import scala.math.log

  // N-gram generation
  def generateNGrams(text: String, n: Int): List[String] = {
    val words = text.toLowerCase.split("\\s+").filter(_.nonEmpty)
    if (words.length < n) List()
    else words.sliding(n).map(_.mkString(" ")).toList
  }

  // Character-level features
  def characterFeatures(text: String): Map[String, Double] = {
    Map(
      "length" -> text.length.toDouble,
      "word_count" -> text.split("\\s+").length.toDouble,
      "avg_word_length" -> {
        val words = text.split("\\s+").filter(_.nonEmpty)
        if (words.nonEmpty) words.map(_.length).sum.toDouble / words.length else 0.0
      },
      "punctuation_ratio" -> text.count(".,!?;:".contains(_)).toDouble / text.length,
      "digit_ratio" -> text.count(_.isDigit).toDouble / text.length,
      "uppercase_ratio" -> text.count(_.isUpper).toDouble / text.length
    )
  }

  // Sentiment lexicon features
  def sentimentFeatures(text: String, positiveWords: Set[String], negativeWords: Set[String]): Map[String, Double] = {
    val words = text.toLowerCase.split("\\s+").toSet
    Map(
      "positive_word_count" -> words.intersect(positiveWords).size.toDouble,
      "negative_word_count" -> words.intersect(negativeWords).size.toDouble,
      "sentiment_ratio" -> {
        val pos = words.intersect(positiveWords).size
        val neg = words.intersect(negativeWords).size
        if (pos + neg > 0) (pos - neg).toDouble / (pos + neg) else 0.0
      }
    )
  }

  // Topic modeling features (simplified LDA)
  def topicFeatures(documents: List[String], numTopics: Int, vocabulary: Set[String]): Array[Array[Double]] = {
    // Simplified topic modeling - in practice, use a proper LDA implementation
    val wordCounts = documents.map { doc =>
      val words = doc.toLowerCase.split("\\s+").filter(vocabulary.contains)
      vocabulary.map(word => words.count(_ == word).toDouble).toArray
    }

    // Random topic assignment for demonstration
    val random = new scala.util.Random(42)
    wordCounts.map { doc =>
      (0 until numTopics).map(_ => random.nextGaussian().abs).toArray
    }.toArray
  }
}
```

### Image Feature Engineering
```scala
object ImageFeatureProcessor {
  // Simulated image processing (in practice, use OpenCV or similar)
  case class Image(width: Int, height: Int, pixels: Array[Array[Double]])

  // Basic statistical features
  def imageStatistics(image: Image): Map[String, Double] = {
    val allPixels = image.pixels.flatten
    Map(
      "mean_intensity" -> allPixels.sum / allPixels.length,
      "std_intensity" -> {
        val mean = allPixels.sum / allPixels.length
        math.sqrt(allPixels.map(p => math.pow(p - mean, 2)).sum / allPixels.length)
      },
      "min_intensity" -> allPixels.min,
      "max_intensity" -> allPixels.max,
      "aspect_ratio" -> image.width.toDouble / image.height
    )
  }

  // Histogram features
  def intensityHistogram(image: Image, numBins: Int = 256): Array[Double] = {
    val allPixels = image.pixels.flatten
    val min = allPixels.min
    val max = allPixels.max
    val binWidth = (max - min) / numBins

    val histogram = Array.fill(numBins)(0.0)
    allPixels.foreach { pixel =>
      val bin = math.min(((pixel - min) / binWidth).toInt, numBins - 1)
      histogram(bin) += 1
    }

    // Normalize
    val total = histogram.sum
    histogram.map(_ / total)
  }

  // Edge detection features (simplified)
  def edgeFeatures(image: Image): Map[String, Double] = {
    val sobelX = Array(Array(-1.0, 0.0, 1.0), Array(-2.0, 0.0, 2.0), Array(-1.0, 0.0, 1.0))
    val sobelY = Array(Array(-1.0, -2.0, -1.0), Array(0.0, 0.0, 0.0), Array(1.0, 2.0, 1.0))

    var edgeStrength = 0.0
    var edgeCount = 0

    for (i <- 1 until image.height - 1; j <- 1 until image.width - 1) {
      val gx = (for (di <- -1 to 1; dj <- -1 to 1) yield
        image.pixels(i + di)(j + dj) * sobelX(di + 1)(dj + 1)).sum
      val gy = (for (di <- -1 to 1; dj <- -1 to 1) yield
        image.pixels(i + di)(j + dj) * sobelY(di + 1)(dj + 1)).sum

      val magnitude = math.sqrt(gx * gx + gy * gy)
      edgeStrength += magnitude
      if (magnitude > 0.1) edgeCount += 1
    }

    Map(
      "edge_density" -> edgeCount.toDouble / ((image.height - 2) * (image.width - 2)),
      "avg_edge_strength" -> edgeStrength / ((image.height - 2) * (image.width - 2))
    )
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


## Phase 7: Performance Optimization and Parallel Processing

### Parallel Collections and Performance
```scala
import scala.collection.parallel.CollectionConverters._
import java.util.concurrent.{ForkJoinPool, TimeUnit}
import scala.concurrent.{Future, Await, ExecutionContext}
import scala.concurrent.duration._

object ParallelProcessing {
  // Custom thread pool for CPU-intensive tasks
  implicit val ec: ExecutionContext = ExecutionContext.fromExecutor(
    new ForkJoinPool(Runtime.getRuntime.availableProcessors())
  )

  // Parallel feature scaling
  def parallelStandardization(features: Array[Array[Double]]): Array[Array[Double]] = {
    features.par.map { row =>
      val mean = row.sum / row.length
      val variance = row.map(x => math.pow(x - mean, 2)).sum / row.length
      val stdDev = math.sqrt(variance)
      row.map(x => if (stdDev > 0) (x - mean) / stdDev else 0.0)
    }.toArray
  }

  // Batch processing for large datasets
  def processBatches[T, U](data: List[T], batchSize: Int, processor: List[T] => List[U]): Future[List[U]] = {
    val batches = data.grouped(batchSize).toList
    val futures = batches.map(batch => Future(processor(batch)))
    Future.sequence(futures).map(_.flatten)
  }

  // Memory-efficient streaming processing
  def streamProcess[T, U](data: Iterator[T], processor: T => U, bufferSize: Int = 1000): Iterator[U] = {
    data.grouped(bufferSize).flatMap(_.par.map(processor).seq)
  }

  // Parallel cross-validation
  def parallelCrossValidation[T](data: Array[T], k: Int, evaluator: (Array[T], Array[T]) => Double): Future[Double] = {
    val folds = createKFolds(data, k)
    val futures = folds.zipWithIndex.map { case (testFold, i) =>
      Future {
        val trainFolds = folds.zipWithIndex.collect { case (fold, j) if j != i => fold }.flatten
        evaluator(trainFolds, testFold)
      }
    }
    Future.sequence(futures).map(scores => scores.sum / scores.length)
  }

  private def createKFolds[T](data: Array[T], k: Int): Array[Array[T]] = {
    val shuffled = scala.util.Random.shuffle(data.toList)
    val foldSize = shuffled.length / k
    shuffled.grouped(foldSize).toArray
  }
}
```

### Memory Management and Optimization
```scala
object MemoryOptimization {
  // Lazy evaluation for large feature pipelines
  class LazyFeaturePipeline[T](data: => Iterator[T]) {
    def map[U](f: T => U): LazyFeaturePipeline[U] = new LazyFeaturePipeline(data.map(f))
    def filter(p: T => Boolean): LazyFeaturePipeline[T] = new LazyFeaturePipeline(data.filter(p))
    def take(n: Int): LazyFeaturePipeline[T] = new LazyFeaturePipeline(data.take(n))
    def collect(): List[T] = data.toList
  }

  // Memory-mapped file processing for very large datasets
  def processLargeFile(filename: String, processor: String => String): Unit = {
    import java.io.{BufferedReader, FileReader, PrintWriter}
    val reader = new BufferedReader(new FileReader(filename))
    val writer = new PrintWriter(s"${filename}_processed")

    try {
      Iterator.continually(reader.readLine())
        .takeWhile(_ != null)
        .foreach(line => writer.println(processor(line)))
    } finally {
      reader.close()
      writer.close()
    }
  }

  // Compression for sparse features
  case class SparseVector(indices: Array[Int], values: Array[Double], length: Int) {
    def toDense: Array[Double] = {
      val dense = Array.fill(length)(0.0)
      indices.zip(values).foreach { case (idx, value) => dense(idx) = value }
      dense
    }

    def dot(other: SparseVector): Double = {
      val thisMap = indices.zip(values).toMap
      other.indices.zip(other.values).map { case (idx, value) =>
        thisMap.getOrElse(idx, 0.0) * value
      }.sum
    }
  }

  def createSparseVector(dense: Array[Double], threshold: Double = 1e-8): SparseVector = {
    val nonZero = dense.zipWithIndex.filter { case (value, _) => math.abs(value) > threshold }
    SparseVector(nonZero.map(_._2), nonZero.map(_._1), dense.length)
  }
}
```

## Phase 8: Testing and Validation Framework

### Property-Based Testing for Feature Engineering
```scala
// Note: In practice, use ScalaCheck for property-based testing
object FeatureTesting {
  import scala.util.Random

  // Property: Standardization should result in mean ≈ 0 and std ≈ 1
  def testStandardizationProperty(features: Array[Double]): Boolean = {
    val standardized = FeatureProcessor.standardize(features)
    val mean = standardized.sum / standardized.length
    val variance = standardized.map(x => math.pow(x - mean, 2)).sum / standardized.length
    val stdDev = math.sqrt(variance)

    math.abs(mean) < 1e-10 && math.abs(stdDev - 1.0) < 1e-10
  }

  // Property: Min-max scaling should result in values between specified bounds
  def testMinMaxScalingProperty(features: Array[Double], newMin: Double, newMax: Double): Boolean = {
    val scaled = FeatureProcessor.minMaxScale(features, newMin, newMax)
    scaled.forall(x => x >= newMin && x <= newMax)
  }

  // Property: One-hot encoding should have exactly one 1 per row
  def testOneHotEncodingProperty(categories: List[String]): Boolean = {
    val encoded = CategoricalEncoder.oneHotEncode(categories)
    encoded.values.forall(_.sum == 1)
  }

  // Data quality checks
  def validateDataQuality(data: Array[Array[Double]]): List[String] = {
    val issues = scala.collection.mutable.ListBuffer[String]()

    // Check for NaN values
    if (data.exists(_.exists(_.isNaN))) {
      issues += "NaN values detected"
    }

    // Check for infinite values
    if (data.exists(_.exists(_.isInfinite))) {
      issues += "Infinite values detected"
    }

    // Check for constant features
    for (colIndex <- data(0).indices) {
      val column = data.map(_(colIndex))
      if (column.distinct.length == 1) {
        issues += s"Constant feature detected at column $colIndex"
      }
    }

    // Check for highly correlated features
    for (i <- data(0).indices; j <- (i + 1) until data(0).length) {
      val corr = pearsonCorrelation(data.map(_(i)), data.map(_(j)))
      if (math.abs(corr) > 0.95) {
        issues += s"High correlation (${corr}) between features $i and $j"
      }
    }

    issues.toList
  }

  private def pearsonCorrelation(x: Array[Double], y: Array[Double]): Double = {
    val n = x.length
    val meanX = x.sum / n
    val meanY = y.sum / n
    val numerator = x.zip(y).map { case (xi, yi) => (xi - meanX) * (yi - meanY) }.sum
    val denomX = math.sqrt(x.map(xi => math.pow(xi - meanX, 2)).sum)
    val denomY = math.sqrt(y.map(yi => math.pow(yi - meanY, 2)).sum)
    if (denomX == 0 || denomY == 0) 0.0 else numerator / (denomX * denomY)
  }

  // Feature importance validation
  def validateFeatureImportance(features: Array[Array[Double]], target: Array[Double]): Map[Int, Double] = {
    features(0).indices.map { i =>
      val feature = features.map(_(i))
      i -> math.abs(pearsonCorrelation(feature, target))
    }.toMap
  }
}
```

## Phase 9: Real-World Case Studies

### Case Study 1: E-commerce Recommendation System
```scala
object EcommerceFeatureEngineering {
  case class Purchase(userId: Int, productId: Int, category: String, price: Double, timestamp: Long)
  case class User(id: Int, age: Int, gender: String, location: String, registrationDate: Long)
  case class Product(id: Int, category: String, brand: String, price: Double, rating: Double)

  def buildUserFeatures(purchases: List[Purchase], users: List[User]): Map[Int, Array[Double]] = {
    val userMap = users.map(u => u.id -> u).toMap

    purchases.groupBy(_.userId).map { case (userId, userPurchases) =>
      val user = userMap(userId)
      val features = Array(
        // Demographic features
        user.age.toDouble,
        if (user.gender == "M") 1.0 else 0.0,
        (System.currentTimeMillis() - user.registrationDate) / (1000 * 60 * 60 * 24).toDouble, // days since registration

        // Behavioral features
        userPurchases.length.toDouble, // total purchases
        userPurchases.map(_.price).sum, // total spent
        userPurchases.map(_.price).sum / userPurchases.length, // average purchase amount
        userPurchases.map(_.category).distinct.length.toDouble, // category diversity

        // Temporal features
        (userPurchases.map(_.timestamp).max - userPurchases.map(_.timestamp).min) / (1000 * 60 * 60 * 24).toDouble, // purchase span in days
        userPurchases.length.toDouble / math.max(1, (System.currentTimeMillis() - user.registrationDate) / (1000 * 60 * 60 * 24)), // purchase frequency

        // Recency features
        (System.currentTimeMillis() - userPurchases.map(_.timestamp).max) / (1000 * 60 * 60 * 24).toDouble // days since last purchase
      )

      userId -> features
    }
  }

  def buildProductFeatures(purchases: List[Purchase], products: List[Product]): Map[Int, Array[Double]] = {
    val productMap = products.map(p => p.id -> p).toMap
    val productPurchases = purchases.groupBy(_.productId)

    products.map { product =>
      val purchaseHistory = productPurchases.getOrElse(product.id, List())
      val features = Array(
        // Product attributes
        product.price,
        product.rating,

        // Popularity features
        purchaseHistory.length.toDouble, // total purchases
        purchaseHistory.map(_.userId).distinct.length.toDouble, // unique buyers
        if (purchaseHistory.nonEmpty) purchaseHistory.map(_.price).sum / purchaseHistory.length else 0.0, // average selling price

        // Category features (simplified one-hot encoding)
        if (product.category == "electronics") 1.0 else 0.0,
        if (product.category == "clothing") 1.0 else 0.0,
        if (product.category == "books") 1.0 else 0.0,

        // Temporal features
        if (purchaseHistory.nonEmpty) {
          (System.currentTimeMillis() - purchaseHistory.map(_.timestamp).max) / (1000 * 60 * 60 * 24).toDouble
        } else Double.MaxValue // days since last purchase
      )

      product.id -> features
    }.toMap
  }
}
```

### Case Study 2: Financial Fraud Detection
```scala
object FraudDetectionFeatures {
  case class Transaction(id: String, userId: String, amount: Double, merchant: String,
                        timestamp: Long, location: String, cardType: String)

  def buildFraudFeatures(transactions: List[Transaction]): Map[String, Array[Double]] = {
    val userTransactions = transactions.groupBy(_.userId)

    transactions.map { transaction =>
      val userHistory = userTransactions(transaction.userId).filter(_.timestamp < transaction.timestamp)

      val features = Array(
        // Amount features
        transaction.amount,
        math.log(transaction.amount + 1), // log-transformed amount

        // User behavior features
        userHistory.length.toDouble, // historical transaction count
        if (userHistory.nonEmpty) userHistory.map(_.amount).sum / userHistory.length else 0.0, // average amount
        if (userHistory.nonEmpty) math.sqrt(userHistory.map(_.amount).map(a => math.pow(a - userHistory.map(_.amount).sum / userHistory.length, 2)).sum / userHistory.length) else 0.0, // amount std dev

        // Deviation from user's typical behavior
        if (userHistory.nonEmpty) {
          val userAvg = userHistory.map(_.amount).sum / userHistory.length
          math.abs(transaction.amount - userAvg) / math.max(userAvg, 1.0)
        } else 0.0,

        // Temporal features
        (transaction.timestamp % (24 * 60 * 60 * 1000)) / (60 * 60 * 1000).toDouble, // hour of day
        ((transaction.timestamp / (24 * 60 * 60 * 1000)) % 7).toDouble, // day of week

        // Location features
        if (userHistory.nonEmpty) {
          val uniqueLocations = userHistory.map(_.location).distinct
          if (uniqueLocations.contains(transaction.location)) 0.0 else 1.0 // new location
        } else 0.0,

        // Merchant features
        if (userHistory.nonEmpty) {
          val uniqueMerchants = userHistory.map(_.merchant).distinct
          if (uniqueMerchants.contains(transaction.merchant)) 0.0 else 1.0 // new merchant
        } else 0.0,

        // Frequency features
        if (userHistory.nonEmpty) {
          val recentTransactions = userHistory.filter(t => transaction.timestamp - t.timestamp < 24 * 60 * 60 * 1000) // last 24 hours
          recentTransactions.length.toDouble
        } else 0.0,

        // Card type encoding
        if (transaction.cardType == "credit") 1.0 else 0.0,
        if (transaction.cardType == "debit") 1.0 else 0.0
      )

      transaction.id -> features
    }.toMap
  }
}
```

## Advanced Learning Resources & Next Steps

### Recommended Study Schedule (Extended):
- **Week 1-2**: Scala basics, syntax, pattern matching, type system
- **Week 3-4**: Collections, functional programming concepts, higher-order functions
- **Week 5-6**: Core feature preprocessing techniques implementation
- **Week 7-8**: Advanced feature engineering (time series, text, images)
- **Week 9-10**: Spark and Breeze integration, parallel processing
- **Week 11-12**: Performance optimization, memory management
- **Week 13-14**: Testing frameworks, validation, real-world projects
- **Week 15-16**: Advanced topics, production deployment considerations

### Practice Exercises (Extended):
1. **Data Pipeline Projects**:
   - Implement end-to-end ETL pipeline with error handling
   - Build real-time feature computation system
   - Create A/B testing framework for feature experiments

2. **Algorithm Implementations**:
   - Custom feature selection algorithms (mutual information, chi-square, LASSO)
   - Dimensionality reduction techniques (PCA, t-SNE, UMAP)
   - Advanced imputation methods (MICE, matrix factorization)

3. **Performance Challenges**:
   - Optimize feature computation for 1M+ records
   - Implement distributed feature engineering with Akka
   - Build streaming feature pipeline with Kafka integration

4. **Domain-Specific Projects**:
   - Computer vision feature extraction pipeline
   - Natural language processing feature engineering
   - Time series forecasting feature creation
   - Graph-based feature engineering for social networks

### Advanced Topics to Explore:
- **Distributed Computing**: Akka for distributed preprocessing, Spark Streaming
- **Functional Libraries**: Cats for advanced functional programming, Shapeless for generic programming
- **Testing**: ScalaTest, ScalaCheck for property-based testing, performance testing
- **Production**: Docker containerization, Kubernetes deployment, monitoring with Prometheus
- **Data Quality**: Great Expectations integration, automated data profiling
- **Feature Stores**: Building feature stores with Delta Lake, feature versioning
- **AutoML**: Automated feature engineering, hyperparameter optimization

### Industry Best Practices:
1. **Feature Engineering Principles**:
   - Domain knowledge incorporation
   - Feature documentation and lineage tracking
   - Reproducible feature pipelines
   - Feature validation and monitoring

2. **Code Quality**:
   - Pure function design for testability
   - Immutable data structures for thread safety
   - Type-driven development for correctness
   - Comprehensive error handling

3. **Performance Optimization**:
   - Lazy evaluation for memory efficiency
   - Parallel processing for CPU-bound tasks
   - Caching strategies for expensive computations
   - Profiling and benchmarking techniques

This comprehensive roadmap provides a structured approach to mastering Scala for ML feature preprocessing, combining theoretical understanding with practical implementation skills. The focus on functional programming principles, performance optimization, and real-world applications prepares you for production-scale feature engineering challenges.