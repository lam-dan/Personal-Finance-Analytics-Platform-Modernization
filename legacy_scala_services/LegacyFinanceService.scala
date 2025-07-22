package com.finance.legacy

import scala.concurrent.{Future, ExecutionContext}
import scala.util.{Success, Failure}
import java.time.LocalDate
import java.math.BigDecimal

/**
 * Legacy Scala Finance Service
 * 
 * This represents the old Scala service that needs to be migrated to Python.
 * It demonstrates common legacy patterns that make migration challenging:
 * - Complex nested data structures
 * - Scala-specific concurrency patterns
 * - Legacy error handling
 * - Performance bottlenecks
 */
class LegacyFinanceService {
  
  // Legacy data structures that are difficult to maintain
  case class LegacyTransaction(
    id: Long,
    amount: BigDecimal,
    category: String,
    date: LocalDate,
    metadata: Map[String, String] = Map.empty
  )
  
  case class LegacyInvestment(
    id: Long,
    asset: String,
    value: BigDecimal,
    lastUpdated: LocalDate,
    riskLevel: String
  )
  
  // Simulated legacy database connection
  private val legacyDb = new LegacyDatabaseConnection()
  
  /**
   * Legacy method with complex error handling and Scala-specific patterns
   * This is the kind of code that needs to be migrated to Python
   */
  def getTransactions(): Future[List[LegacyTransaction]] = {
    try {
      // Legacy database query with complex error handling
      val result = legacyDb.query("SELECT * FROM legacy_transactions")
      
      // Scala-specific data transformation
      val transactions = result.map { row =>
        LegacyTransaction(
          id = row.getLong("id"),
          amount = new BigDecimal(row.getString("amount")),
          category = row.getString("category"),
          date = LocalDate.parse(row.getString("date")),
          metadata = parseMetadata(row.getString("metadata"))
        )
      }.toList
      
      Future.successful(transactions)
    } catch {
      case e: Exception =>
        // Legacy error handling pattern
        println(s"Error fetching transactions: ${e.getMessage}")
        Future.failed(e)
    }
  }
  
  /**
   * Legacy method with performance issues that need optimization
   */
  def getInvestments(): Future[List[LegacyInvestment]] = {
    // Simulate legacy performance bottleneck
    Thread.sleep(500) // Legacy blocking operation
    
    val investments = List(
      LegacyInvestment(1L, "Stocks", new BigDecimal("15000.00"), LocalDate.now(), "HIGH"),
      LegacyInvestment(2L, "Bonds", new BigDecimal("5000.00"), LocalDate.now(), "LOW")
    )
    
    Future.successful(investments)
  }
  
  /**
   * Complex legacy business logic that needs to be reimplemented in Python
   */
  def calculateSpendingTrends(startDate: LocalDate, endDate: LocalDate): Future[Map[String, BigDecimal]] = {
    getTransactions().map { transactions =>
      val filtered = transactions.filter { tx =>
        tx.date.isAfter(startDate.minusDays(1)) && tx.date.isBefore(endDate.plusDays(1))
      }
      
      // Legacy complex aggregation logic
      filtered.groupBy(_.category).mapValues { txs =>
        txs.foldLeft(BigDecimal.ZERO) { (sum, tx) =>
          sum.add(tx.amount)
        }
      }
    }
  }
  
  // Legacy helper methods
  private def parseMetadata(metadataStr: String): Map[String, String] = {
    if (metadataStr == null || metadataStr.isEmpty) {
      Map.empty
    } else {
      metadataStr.split(",").map { pair =>
        val parts = pair.split(":")
        if (parts.length == 2) {
          parts(0) -> parts(1)
        } else {
          "" -> ""
        }
      }.toMap
    }
  }
}

// Simulated legacy database connection
class LegacyDatabaseConnection {
  def query(sql: String): List[Map[String, Any]] = {
    // Simulate legacy database response
    List(
      Map("id" -> 1L, "amount" -> "-120.50", "category" -> "Groceries", "date" -> "2024-07-01", "metadata" -> ""),
      Map("id" -> 2L, "amount" -> "-75.20", "category" -> "Transportation", "date" -> "2024-07-03", "metadata" -> ""),
      Map("id" -> 3L, "amount" -> "5000.00", "category" -> "Salary", "date" -> "2024-07-05", "metadata" -> "")
    )
  }
} 