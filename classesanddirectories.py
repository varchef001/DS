class Student:
    def __init__(self, name, rollnumber, marks):
        self.name = name
        self.rollnumber = rollnumber
        self.marks = marks
    def displaydetails(self):
        print(self.name, self.rollnumber, self.marks, self.ispassed())
    def ispassed(self):
        if self.marks >= 40:
            return "Pass"
        else:
            return "Fail"
    def getmarks(self):
        return self.marks
    def setmarks(self, newmarks):
        if newmarks >= 0 and newmarks <= 100:
            self.marks = newmarks
        else:
            print("Invalid marks")
student1 = Student("Alice", 101, 85)
student2 = Student("Bob", 102, 35)
student3 = Student("Charlie", 103, 60)
students = [student1, student2, student3]
for student in students:
    student.displaydetails()
class BankAccount:
    bankname = "ABC Bank"
    def __init__(self, accountholder, accountnumber, balance):
        self.accountholder = accountholder
        self.accountnumber = accountnumber
        self.balance = balance
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
        else:
            print("Invalid deposit")
    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
        else:
            print("Insufficient balance")
    def displaybalance(self):
        print(BankAccount.bankname, self.accountholder, self.accountnumber, self.balance)
account1 = BankAccount("David", 123456, 5000)
account1.displaybalance()
account1.deposit(1500)
account1.withdraw(2000)
account1.displaybalance()