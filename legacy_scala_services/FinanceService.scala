object FinanceService extends App {
  // Define the Transaction case class
  case class Transaction(id: Int, amount: Double, category: String, date: String)

  // Define the Investment case class
  case class Investment(id: Int, asset: String, value: Double, lastUpdated: String)

  // Sample transactions data
  val transactions = List(
    Transaction(1, -120.50, "Groceries", "2024-07-01"),
    Transaction(2, -75.20, "Transportation", "2024-07-03")
  )

  // Sample investments data
  val investments = List(
    Investment(1, "Stocks", 15000, "2024-07-07"),
    Investment(2, "Bonds", 5000, "2024-07-07")
  )

  // Print all transactions
  println("Transactions:")
  transactions.foreach(println)

  // Print all investments
  println("Investments:")
  investments.foreach(println)
}