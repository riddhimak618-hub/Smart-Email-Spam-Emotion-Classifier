import pandas as pd, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("="*55)
print("  Smart Email Spam & Emotion Classifier")
print("  Model Training...")
print("="*55)

# ── Load CSV dataset ──────────────────────────────────────────
CSV_PATH = "spam_email_dataset.csv"

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    df.columns = ["Subject", "Body", "label"]
    df = df.dropna(subset=["Subject", "Body", "label"])
    df["label"] = df["label"].astype(int)
    print(f"\n[INFO] CSV loaded: {CSV_PATH}")
else:
    # Fallback: hardcoded dataset (agar CSV na mile toh)
    print(f"\n[WARN] '{CSV_PATH}' not found — using built-in dataset.")
    DATASET = [
        ("WINNER Claim Your Prize NOW","Congratulations You have been selected as our lucky winner Click here to claim your cash prize instantly Act fast offer expires in 24 hours",1),
        ("FREE GIFT CARD You have been chosen","You are eligible for a FREE Amazon gift card Limited time offer Click the link below and enter your details to claim before it expires tonight",1),
        ("Make 5000 Per Day Working From Home","Our proven system lets anyone earn 5000 per day from home No experience needed Join thousands of happy members Sign up now and start earning today",1),
        ("URGENT Your account will be suspended","We have detected suspicious activity on your account You must verify your identity immediately by clicking the link below or your account will be permanently suspended",1),
        ("Congratulations You Won the Lottery","Dear winner your email has been randomly selected in our international lottery You have won 2500000 To claim your prize send us your full name address and phone number",1),
        ("Your Computer Has a Virus Fix It Now","Alert Your computer has been infected with a dangerous virus Click below to run a free security scan immediately Your data is at risk Download our FREE antivirus tool now",1),
        ("Nigerian Prince Needs Your Help","Dear friend I am a Nigerian prince with 10 million I need to transfer abroad I will give you 30 percent if you help me Please reply with your bank details to proceed",1),
        ("Double Your Bitcoin in 24 Hours","Our exclusive trading bot guarantees 200 percent returns on your Bitcoin investment within 24 hours Join 50000 happy investors Send BTC now and receive double tomorrow",1),
        ("Special Offer Lose 30 Pounds in 30 Days","Our revolutionary weight loss pill melts fat overnight with zero exercise Buy 1 bottle get 2 FREE Limited stock Order within the next 10 minutes for 90 percent off",1),
        ("You owe money to IRS Call Immediately","This is a final notice from IRS You owe 3450 in back taxes You must call us immediately to avoid arrest Failure to respond will result in legal action",1),
        ("FINAL NOTICE Claim Your Government Grant","You have been pre-approved for a 7500 government grant that never needs to be repaid This is your final notice Claim now at our website No credit check required",1),
        ("Hot Singles In Your Area Are Waiting","12 attractive singles near you are waiting to meet you tonight Sign up FREE and start chatting now No credit card needed Join millions of happy members today",1),
        ("Earn 1000 per Day Secret Method","Wall Street insiders dont want you to know this secret money making trick I went from broke to 1000 a day in just 2 weeks using this simple method Click to learn more",1),
        ("Pre-Approved Loan No Credit Check","You have been pre-approved for a 50000 personal loan regardless of your credit history No documents needed Money transferred directly to your account within hours Apply now",1),
        ("WIN an iPhone Lucky Visitor","Congratulations You are todays lucky visitor You have been selected to win an iPhone Complete a short survey to claim your prize Only 3 spots remaining",1),
        ("ALERT Suspicious Login Verify Account","We detected a login to your account from an unknown device in Russia If this was not you click here immediately to secure your account and reset your password",1),
        ("You Have Unclaimed Funds Waiting","Our records show you have 4231 in unclaimed funds This money will be forfeited if not claimed within 48 hours Click here to verify your identity and claim your money",1),
        ("Free Trial Just Pay Shipping","Try our premium supplement FREE for 30 days Just pay shipping If you love it we will charge 89 per month Cancel anytime online Order today",1),
        ("Make Money Online From Home","Single mom discovers simple trick to make 300 per hour from home on her laptop No experience required Learn her secret method now Join free today",1),
        ("Cheap Prescription Drugs No Prescription","Buy any prescription medication without a prescription at 90 percent off Fast worldwide shipping Order securely online today",1),
        ("URGENT Production Server Down","Hi team the production server went down at 2 AM Customers cannot access the platform I need the DevOps team to respond immediately This is a critical outage affecting all users",0),
        ("Critical Security Breach Detected","All staff must change their passwords immediately Our security team has detected unauthorized access to the employee portal Please log out of all devices and reset your password right away",0),
        ("Action Required Submit Timesheet Today","This is a reminder that payroll closes today at 5 PM If you have not submitted your timesheet for this week you will not receive payment on Friday Please submit immediately",0),
        ("URGENT Client Presentation Slides Missing","The client arrives at 3 PM and the final slides are not in the shared drive I need whoever has the latest version to upload it right now This cannot wait Please respond ASAP",0),
        ("Contract Renewal Deadline is Tomorrow","This is your final reminder that the vendor contract expires tomorrow at midnight If we do not renew by then all services will be terminated Please approve the renewal today",0),
        ("This Service is Completely Unacceptable","I am writing to express my extreme frustration with your company This is the third time my order has been delayed and no one has bothered to contact me I demand a full refund immediately",0),
        ("Formal Complaint Worst Experience Ever","I have never encountered such incompetent and disgraceful service in my life Your staff was rude dismissive and completely unhelpful I will be escalating this to your management",0),
        ("I Am Furious About My Missing Delivery","It has been 3 weeks since I placed my order and it still has not arrived Every time I call your support team they give me a different story This is absolutely ridiculous I want this resolved TODAY",0),
        ("Unacceptable Billing Error Charged Twice","You have charged my credit card twice for the same order I reported this 10 days ago and nothing has been done I am extremely angry about this Fix this immediately",0),
        ("Congratulations on Your Promotion","Dear Sarah I am absolutely thrilled to share the wonderful news of your well-deserved promotion to Senior Manager Your hard work and dedication have truly made a difference Congratulations",0),
        ("Thank You Outstanding Project Delivery","Hi team I just wanted to say how incredibly proud I am of everyones effort on the project The client called this morning and they are absolutely delighted Fantastic work",0),
        ("Great News We Won the Contract","Everyone please join me in celebrating some amazing news We have just been awarded the 2 million contract with ABC Corporation This is a huge win for our team Well done",0),
        ("Your Application Has Been Accepted","Dear Michael I am delighted to inform you that your application for the MBA program has been accepted You were selected from over 2000 applicants Congratulations and welcome",0),
        ("Unfortunately We Cannot Offer You the Position","Dear James thank you for interviewing with us After careful consideration we regret to inform you that we have decided to move forward with another candidate Best wishes",0),
        ("Sorry for the Inconvenience","We sincerely apologize for the service disruption you experienced yesterday We understand how frustrating this must have been and we are truly sorry for the impact on your work",0),
        ("Meeting Notes Sprint Review","Hi team please find the meeting notes from yesterdays sprint review attached We covered progress on feature X Q2 timeline and resource allocation Next meeting is Monday at 10 AM",0),
        ("Q3 Financial Report Attached","Please find attached the Q3 financial report for your review Total revenue came in at 4.2 million against a target of 4.0 million Operating expenses were within budget",0),
        ("Team Lunch This Friday at 1 PM","Just a reminder that the team lunch is confirmed for this Friday at 1 PM at The Grand Bistro Please let me know by Wednesday if you cannot make it",0),
        ("Office Will Be Closed on Monday","Please note that the office will be closed on Monday due to the public holiday Normal operations will resume on Tuesday morning Contact the on-call team for urgent matters",0),
        ("Project Proposal Submitted","Hi John just following up to confirm that I submitted the project proposal to the client this morning I also sent them the supporting documents I expect to hear back by end of week",0),
        ("Interview Scheduled for Tuesday","Dear Applicant we are pleased to confirm your interview has been scheduled for Tuesday at 2 PM at our Head Office Please bring a copy of your CV and any relevant certificates",0),
    ]
    df = pd.DataFrame(DATASET, columns=["Subject", "Body", "label"])

# ── Combine Subject + Body ────────────────────────────────────
df["text"] = df["Subject"].fillna("") + " " + df["Body"].fillna("")

print(f"\nDataset: {len(df)} emails | Spam: {df.label.sum()} | Ham: {(df.label==0).sum()}")

# ── Train/Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ── Build Pipeline ────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))),
    ("clf",   MultinomialNB(alpha=0.1))
])

# ── Train ─────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ── Evaluation ────────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {acc * 100:.1f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham (Not Spam)", "Spam"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"               Predicted Ham  Predicted Spam")
print(f"Actual Ham          {cm[0][0]:<14} {cm[0][1]}")
print(f"Actual Spam         {cm[1][0]:<14} {cm[1][1]}")

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\nPrecision : {precision*100:.1f}%  (kitne predicted spam actually spam the)")
print(f"Recall    : {recall*100:.1f}%  (kitne actual spam pakde gaye)")

# ── Save Model ────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
with open("model/spam_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModel saved to model/spam_model.pkl")
print("\nNow run: python app.py")
print("="*55)
