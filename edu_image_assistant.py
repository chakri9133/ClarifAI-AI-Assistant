import easyocr #type: ignore
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import requests
import json

# Load OCR and captioning
reader = easyocr.Reader(['en'], gpu=False)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()

# Load topic classifier
topic_model = SentenceTransformer("all-MiniLM-L6-v2")
topics = [
    "mathematics", "physics", "chemistry", "biology", "computer science",
    "history", "geography", "english", "political science", "economics",
    "environment", "technology", "art", # NCERT: Primary to Secondary
    "place value", "shapes", "multiplication", "division", "area and perimeter", "environment", "types of plants", "water cycle",
    "solar system", "motion and measurement", "light and shadow", "pollution", "reproduction in animals", "nutrition in plants",
    "democracy", "indian constitution", "judiciary", "natural resources", "tribal communities", "social reformers",
    # Middle School
    "integers", "fractions", "decimals", "geometry basics", "algebra basics", "data handling", "simple equations",
    "basic statistics", "introduction to science", "states of matter", "force and motion", "work and energy", "ecosystems",
    "human body systems", "health and hygiene", "environmental conservation", "civics basics", "geography basics",
    "history basics", "cultural heritage", "arts and crafts", "music and dance", "sports and games", "life skills education",   
    # Secondary School
    "polynomials", "quadratic equations", "trigonometry basics", "statistics and probability", "linear equations in two variables",
    "geometry theorems", "mensuration basics", "coordinate geometry", "chemical equations", "acids and bases", "metals and non-metals",
    "light and optics", "electricity basics", "magnetism", "human reproduction", "heredity and evolution", "ecosystem dynamics",
    "social science concepts", "geographical features", "historical events", "civics and governance", "economic basics",
    "environmental science", "disaster management", "entrepreneurship education", "financial literacy", "digital literacy",
    "communication skills", "critical thinking", "problem solving", "teamwork and collaboration", "creativity and innovation",
    "emotional intelligence", "leadership skills", "time management", "goal setting", "self-awareness", "adaptability",
    "conflict resolution", "decision making", "ethical reasoning", "cultural awareness", "global citizenship", "sustainability education",
    "research skills", "information literacy", "media literacy", "study skills", "career exploration", "workplace readiness",

    # High School
    "quadrilaterals", "mensuration", "linear equations", "algebraic identities", "coordinate geometry", "probability",
    "chemical reactions", "periodic classification", "carbon compounds", "electricity and circuits", "magnetic effects",
    "life processes", "control and coordination", "reaching adolescence", "our environment", "diversity in organisms",
    "motion in a straight line", "motion in a plane", "work energy and power", "gravitation", "waves and sound",
    "light waves", "heat and temperature", "chemical bonding", "redox reactions", "hydrocarbons", "environmental chemistry",
    "statistics and probability", "linear programming", "conic sections", "three-dimensional geometry", "matrices and determinants",
    "vectors", "complex numbers", "trigonometric ratios", "trigonometric identities", "inverse trigonometric functions",
    "statistics", "probability distributions", "data interpretation", "financial mathematics", "linear inequalities",
    "permutations and combinations", "binomial theorem", "sequence and series", "limits and derivatives", "integrals",
    "differential equations", "conic sections", "three-dimensional geometry", "linear programming", "matrices and determinants",
    "complex numbers", "vectors", "statistics and probability", "mathematical reasoning", "mathematical induction",

    # Intermediate & State Board
    "integration", "differentiation", "vectors", "complex numbers", "probability distributions", "mechanics of solids",
    "cell biology", "plant physiology", "biomolecules", "classification of elements", "electrochemistry",
    "business organization", "public finance", "current affairs",
    "indian polity", "indian economy", "world geography", "world history", "environmental science", "disaster management",
    "entrepreneurship development", "financial literacy", "digital marketing basics", "data interpretation skills",
    "communication skills", "critical thinking skills", "problem solving skills", "teamwork and collaboration skills",
    "creativity and innovation skills", "emotional intelligence skills", "leadership skills", "time management skills",
    "goal setting skills", "self-awareness skills", "adaptability skills", "conflict resolution skills", "decision making skills",
    "ethical reasoning skills", "cultural awareness skills", "global citizenship skills", "sustainability education skills",


    # Engineering & Diploma
    "strength of materials", "surveying", "machine drawing", "industrial management", "thermal engineering",
    "network analysis", "analog circuits", "microcontrollers", "PLC systems", "database design", "web development",
    "java programming", "python programming", "c programming", "oop concepts", "file handling in c",
    "data structures in c", "algorithms in c", "operating systems concepts", "computer networks basics",
    "database management systems", "software engineering principles", "web technologies basics", "mobile app development",
    "cloud computing concepts", "cyber security basics", "machine learning fundamentals", "artificial intelligence basics",
    "data science fundamentals", "big data technologies", "internet of things basics", "blockchain technology basics",
    "agile methodology", "devops practices", "version control systems", "api development", "responsive web design",
    "cross platform app development", "ui/ux principles", "software testing techniques", "ethical hacking basics",
    "penetration testing basics", "network security fundamentals", "data privacy laws", "digital marketing strategies",
    "search engine optimization techniques", "content management systems", "e-commerce platforms", "social media marketing",
    # College Level
    "advanced calculus", "linear algebra", "discrete mathematics", "numerical methods", "graph theory",
    "combinatorics", "complex analysis", "real analysis", "abstract algebra", "differential equations",
    "partial differential equations", "functional analysis", "topology", "mathematical logic", "set theory",
    "optimization techniques", "operations research", "game theory", "cryptography", "information theory",
    "machine learning algorithms", "deep learning techniques", "natural language processing", "computer vision",
    "reinforcement learning", "neural networks", "fuzzy logic", "genetic algorithms", "swarm intelligence",
    "cloud computing architectures", "distributed systems", "parallel computing", "high performance computing",
    "data mining techniques", "big data analytics", "data visualization techniques", "business intelligence tools",
    "data warehousing concepts", "data governance practices", "data ethics and privacy", "data security measures",
    "data quality assurance", "data lineage tracking", "data cataloging practices", "data integration techniques",
    "data architecture design", "data modeling techniques", "data visualization tools", "business intelligence strategies",
    "dashboard creation techniques", "sql for data analysis", "python for data science", "r programming basics",
    "machine learning algorithms", "deep learning basics", "neural network architectures", "convolutional neural networks",
    "recurrent neural networks", "natural language understanding", "chatbot development", "image processing techniques",
    "computer vision techniques", "audio signal processing", "time series forecasting", "anomaly detection techniques",

    # Advanced Tech & Programming
    "lambda functions", "regex in python", "inheritance in java", "multithreading", "promises js", "event loop js",
    "sql query to find second highest salary", "joins", "stored procedures", "nosql", "mongodb queries",
    "devops basics", "docker", "rest apis", "http status codes", "git branches", "merge conflict resolution",
    "android studio", "react native basics",
    

    # Competitive Programming
    "linked lists", "binary trees", "recursion", "dp problems", "graph traversal", "stack queue", "search algorithms",
    "leetcode problems", "hackerrank practice", "geeksforgeeks solutions", "codeforces contests", "competitive programming strategies",
    "algorithm design", "data structure optimization", "time complexity analysis", "space complexity analysis",
    # Career & Job Preparation
    "resume writing tips", "interview preparation", "soft skills development", "job search strategies", "networking techniques",
    "personal branding", "career planning", "professional development", "time management skills", "goal setting techniques",
    "self-awareness exercises", "adaptability skills", "conflict resolution techniques", "decision making strategies",
    "ethical reasoning skills", "cultural awareness techniques", "global citizenship education", "sustainability practices",
    "research skills development", "information literacy techniques", "media literacy education", "study skills improvement",
    "career exploration techniques", "workplace readiness skills", "entrepreneurship education", "financial literacy basics",
    "digital marketing strategies", "data interpretation skills", "communication skills development", "critical thinking exercises",
    "problem solving techniques", "teamwork and collaboration skills", "creativity and innovation exercises", "emotional intelligence development",
    "leadership skills development", "time management techniques", "goal setting strategies", "self-awareness practices",
    "adaptability exercises", "conflict resolution skills", "decision making techniques", "ethical reasoning practices",
    "cultural awareness education", "global citizenship skills", "sustainability education practices", "research skills techniques",
    "information literacy skills", "media literacy techniques", "study skills strategies", "career exploration practices",
    "workplace readiness techniques", "entrepreneurship skills", "financial literacy education", "digital marketing basics",
    "data interpretation techniques", "communication skills exercises", "critical thinking skills", "problem solving practices",
    "teamwork and collaboration techniques", "creativity and innovation skills", "emotional intelligence exercises",
    "leadership skills techniques", "time management practices", "goal setting exercises", "self-awareness skills",
    "adaptability techniques", "conflict resolution practices", "decision making skills", "ethical reasoning techniques",
    # Data Science & Analytics
    "data visualization", "pandas basics", "numpy arrays", "matplotlib plots", "seaborn visualizations",
    "data cleaning", "exploratory data analysis", "feature selection", "time series analysis", "predictive modeling",
    "classification algorithms", "regression algorithms", "clustering techniques", "dimensionality reduction",
    "natural language processing", "text mining", "sentiment analysis", "recommendation systems", "big data technologies",
    "spark basics", "hadoop ecosystem", "data warehousing", "etl processes", "data governance", "data ethics",
    "data privacy", "data security", "data quality", "data lineage", "data cataloging", "data integration",
    "data architecture", "data modeling", "data visualization tools", "business intelligence", "dashboard creation",
    "sql for data analysis", "python for data science", "r programming", "machine learning algorithms",
    "deep learning basics", "neural network architectures", "convolutional neural networks", "recurrent neural networks",
    "natural language understanding", "chatbot development", "image processing", "computer vision techniques",
    "audio signal processing", "time series forecasting", "anomaly detection", "data storytelling", "data-driven decision making",
    "data ethics and bias", "explainable ai", "model interpretability", "model deployment", "model monitoring",
    "model versioning", "model retraining", "cloud data services", "data pipelines", "data lakes", "data marts",
    "data mesh architecture", "data fabric", "data virtualization", "dataOps practices", "agile data development",
    "data science project lifecycle", "data science methodologies", "data science frameworks", "data science tools",
    
     # Machine Learning & AI Models
     "underfitting", "overfitting", "cross-validation", "hyperparameter tuning", "feature scaling",
    "feature selection techniques", "model evaluation metrics", "confusion matrix", "precision recall f1 score",
    "accuracy vs precision", "roc auc curve", "precision recall curve", "classification report", "regression metrics",
    "mean squared error", "root mean squared error", "r squared", "adjusted r squared", "cross entropy loss",
    "logistic regression", "linear regression", "decision trees", "random forest", "svm", "k-means clustering",
    "k-nearest neighbors", "naive bayes", "neural networks", "cnn", "rnn", "lstm", "transformer architecture",
    "gradient descent", "loss functions", "bias variance tradeoff", "feature engineering", "data preprocessing",
    "model evaluation metrics", "confusion matrix", "roc curve", "sklearn usage", "tensorflow basics", "pytorch code",
    "unsupervised learning", "reinforcement learning", "transfer learning", "llm fine tuning", "openai api",
    "huggingface transformers", "nlp techniques", "text classification", "named entity recognition", "language models",
    "chatbot frameworks", "speech recognition", "image classification", "object detection", "face recognition",
    "time series forecasting", "anomaly detection", "recommendation algorithms", "collaborative filtering",
    "content-based filtering", "matrix factorization", "deep learning frameworks", "model deployment strategies",
    "model monitoring tools", "model interpretability techniques", "explainable ai methods", "data augmentation",
    "hyperparameter tuning", "cross-validation techniques", "grid search", "random search", "bayesian optimization",
    "ensemble methods", "bagging", "boosting", "stacking", "model compression", "quantization techniques",
    "knowledge graphs", "graph neural networks", "reinforcement learning algorithms", "multi-agent systems",
    "federated learning", "differential privacy", "adversarial attacks", "model robustness", "ethical ai practices",
    "ai governance", "ai ethics", "ai bias mitigation", "ai explainability", "ai fairness", "ai transparency",
    "ai accountability", "ai regulation", "ai standards", "ai certification", "ai risk management", "ai compliance",
    "ai security", "ai privacy", "ai trustworthiness", "ai reliability", "ai performance metrics", "ai benchmarking",
    "ai scalability", "ai deployment challenges", "ai integration", "ai in production", "ai monitoring",
    "ai maintenance", "ai lifecycle management", "ai project management", "ai team collaboration", "ai communication skills",
    
    # General Knowledge & Civics
    "fundamental rights", "duties of citizens", "national symbols", "election process in india", "rti act",
    "73rd constitutional amendment", "freedom fighters of india",
    "indian geography basics", "world geography basics", "current affairs", "indian economy basics",
    "indian history basics", "world history basics", "indian culture and heritage", "global issues", "international organizations",
    "united nations", "world health organization", "international monetary fund", "world bank", "nato", "g20 summit",
    "climate change", "sustainable development goals", "globalization", "digital divide", "cybersecurity issues",
    "human rights", "Capital of India", "Indian Constitution", "Preamble of the Constitution", "Fundamental Rights",
    "Fundamental Duties", "Directive Principles of State Policy", "Separation of Powers", "Judiciary System in India",
    "Parliamentary System", "President of India", "Prime Minister of India", "State Governments", "Local Self-Government",
    "Elections in India", "Political Parties", "Election Commission of India", "Constitutional Amendments",
    "Indian Independence Movement", "Freedom Fighters", "Constitutional Development", "Post-Independence India",
    "Indian Economy", "Economic Planning", "Five-Year Plans", "Monetary Policy", "Fiscal Policy", "Public Finance", 
    # Language & Literature
    "figures of speech", "essay writing format", "note making", "reading comprehension", "poetry analysis",
    "shakespeare drama", "indian literature", "tagore poems", "english grammar mcq", "precis writing",
    "father of Nation", "national anthem", "national song", "indian languages", "english literature basics",
    "world literature", "literary devices", "poetic forms", "prose analysis", "drama techniques", "narrative styles",
    "Mahathma Gandhi", "Jawaharlal Nehru", "B.R. Ambedkar", "Sardar Patel", "Subhas Chandra Bose",
    "Rani Lakshmibai", "Bhagat Singh", "Dr. B.R. Ambedkar", "Sarojini Naidu", "Vivekananda", "Tagore",
    "Kalidasa", "Tulsidas", "Mirabai", "Kabir", "Faiz Ahmed Faiz", "Ghalib", "Premchand", "Ismat Chughtai",
    # Science & Technology
    "scientific method", "basic physics concepts", "chemistry basics", "biology fundamentals", "environmental science",
    
    # Lab Practicals & Experiments
    "vernier caliper reading", "screw gauge usage", "biology slide preparation", "chemistry viva questions",
    "physics practical experiments", "ubuntu terminal commands",
    # Educational Pedagogy
    "constructivist learning", "active learning techniques", "differentiated instruction", "formative assessment",
    "summative assessment", "project-based learning", "inquiry-based learning", "cooperative learning",
    "blended learning", "flipped classroom", "gamification in education", "learning styles", "multiple intelligences",
    "cognitive development", "piaget's theory", "vygotsky's theory", "bloom's taxonomy", "andragogy vs pedagogy",
    "learning outcomes", "curriculum design", "lesson planning", "classroom management",


    # Pedagogy & Teaching Exams
    "lesson planning", "bloom taxonomy", "formative vs summative assessment", "piaget theory",
    "constructivist approach", "educational psychology", "nep 2020 summary",

    # Applied Math & Statistics
    "z score", "t score", "anova", "binomial distribution", "poisson distribution", "normal distribution",
    "real life statistics", "sampling techniques", "inferential statistics", "vedic math tricks",
    "mathematical induction", "set theory basics", "logic gates", "boolean algebra", "graph theory basics",
    "linear programming", "calculus basics", "differential equations", "numerical methods", "matrix operations",
    "vector calculus", "complex numbers", "fourier series", "laplace transforms", "statistics basics",
    "probability theory", "game theory basics", "fuzzy logic", "chaos theory", "fractal geometry", "number theory basics",
    "combinatorial mathematics", "graph algorithms", "dynamic programming", "greedy algorithms", "divide and conquer",
    "backtracking algorithms", "string algorithms", "bit manipulation techniques", "computational geometry",
    "cryptography basics", "error detection and correction", "hashing techniques", "data compression algorithms",

    #Charted Accountant
    "accounting principles", "financial statements", "cost accounting", "taxation basics", "auditing standards",
    "financial management", "corporate law", "indirect taxation", "direct taxation", "gst basics", "income tax laws",
    "company law", "securities law", "financial reporting", "international accounting standards", "forensic accounting",
    "valuation techniques", "mergers and acquisitions", "financial analysis", "budgeting and forecasting",
    "risk management", "internal controls", "compliance and governance", "financial modeling", "investment analysis",
    "capital markets", "derivatives trading", "portfolio management", "financial instruments", "financial regulations",
    "financial ethics", "corporate social responsibility", "sustainability reporting", "financial literacy",
    "financial planning", "retirement planning", "estate planning", "tax planning", "wealth management",
    "financial advisory", "business valuation", "financial statement analysis", "cost control techniques",
    "budgeting techniques", "financial forecasting", "cash flow management", "working capital management",
    "capital budgeting", "financial ratios", "performance measurement", "financial risk assessment", "investment strategies",
    "portfolio diversification", "asset allocation", "financial instruments analysis", "equity research", "fixed income analysis",
    "derivatives analysis", "alternative investments", "real estate investment", "private equity", "venture capital",
    "financial technology", "blockchain in finance", "cryptocurrency basics", "robo-advisory services", "crowdfunding",
    # Competitive Exams & Aptitude
    "quantitative aptitude", "logical reasoning", "verbal ability", "data interpretation", "general awareness",
    "current affairs quiz", "banking awareness", "insurance basics", "ssc cgl preparation", "railway exam tips",
    "upsc prelims strategy", "state psc exam tips", "cat exam preparation", "gate exam syllabus", "iit jee preparation",
    "neet exam tips", "jee mains preparation", "cbse board exam tips", "icse board exam tips", "state board exam tips",
    "ssc chsl exam tips", "ibps po preparation", "rrb ntpc exam tips", "clat exam syllabus", "ctet exam preparation",
    "net/jrf exam tips", "ugc net syllabus", "set exam preparation", "ctet exam syllabus", "stet exam tips",
    "ctet exam pattern", "stet exam syllabus", "ctet exam preparation", "stet exam pattern", "ctet exam tips",
    "ssc mts exam tips", "ibps clerk preparation", "rrb group d exam tips", "railway group d preparation",
    "upsc mains strategy", "state psc exam syllabus", "cat exam syllabus", "gate exam preparation tips",
    "iit jee syllabus", "neet exam preparation tips", "jee mains syllabus", "cbse board exam syllabus",
    
    # Computer Science & IT
    "data structures", "algorithms", "operating systems", "computer networks", "database management",
    "software engineering", "web technologies", "mobile app development", "cloud computing", "cyber security",
    "machine learning", "artificial intelligence", "data science", "big data", "internet of things",
    "blockchain technology", "agile methodology", "devops practices", "version control systems", "api development",
    "responsive web design", "cross platform app development", "ui/ux principles", "software testing",
    "ethical hacking", "penetration testing", "network security", "data privacy laws", "digital marketing",
    "search engine optimization", "content management systems", "e-commerce platforms", "social media marketing",
    "HTTP","HTTPS", "REST APIs", "GraphQL", "WebSockets", "AJAX", "JSON", "XML", "HTML5", "CSS3",
    "JavaScript", "TypeScript", "React.js", "Angular.js", "Vue.js", "Node.js", "Express.js", "Django",
    "Flask", "Ruby on Rails", "Spring Boot", "ASP.NET Core", "Laravel", "WordPress", "Magento",

    # Soft Skills
    "resume writing", "public speaking", "group discussion", "pomodoro technique", "interview etiquette",
    "eisenhower matrix", "logical reasoning", "numerical ability",
    "critical thinking", "time management", "conflict resolution", "team collaboration", "emotional intelligence",
    "adaptability", "problem solving", "decision making", "creativity", "negotiation skills", "presentation skills",
    "active listening", "assertiveness", "networking skills", "cultural awareness", "feedback techniques",
    "mentoring and coaching", "self-motivation", "goal setting", "stress management", "work-life balance",
    "leadership qualities", "customer service skills", "sales techniques", "persuasion skills", "influence strategies",
    "cross-cultural communication", "digital literacy", "online etiquette", "social skills", "empathy in communication",
    "conflict management", "team dynamics", "collaborative problem solving", "creative thinking techniques",
    "brainstorming methods", "mind mapping", "design thinking", "agile project management", "scrum methodology",
    "kanban system", "lean principles", "six sigma basics", "quality assurance", "risk management",
    "strategic planning", "business analysis", "market research", "customer relationship management",
    "financial literacy", "budgeting skills", "investment basics", "entrepreneurial mindset", "innovation strategies",
    "change management", "organizational behavior", "workplace ethics", "corporate social responsibility",
    "employer expectations", "job market trends", "career development", "professional networking", "personal branding",


    # Misc
    "communication skills", "research methodology", "entrepreneurship basics", "startup ecosystem",
    "how many apples", "primary math", "simple arithmetic", "number story", "story sums",
    "number systems", "lcm and hcf", "simplification", "surds and indices", "fractions and decimals", "ratio and proportion", "averages", "percentages",
    "profit and loss", "simple interest and compound interest", "time and work", "pipes and cisterns", "time speed and distance",
    "boats and streams", "mixtures and alligation", "partnership problems", "problems on ages", "mensuration", "permutations and combinations", "probability", "data interpretation",
    "profit and loss", "discount", "simple interest", "compound interest", "time and work", "work and wages", "pipes and cisterns", "time speed and distance",
    "boats and streams", "mixtures and alligation", "partnership problems", "problems on ages", "mensuration", "permutations and combinations", "probability", "data interpretation",
    "algebra equations", "quadratic equations", "basic geometry", "basic trigonometry", "coding and decoding", "blood relations", "direction sense", "syllogisms",
    "number series", "letter series", "analogy problems", "classification reasoning", "seating arrangement", "logical puzzles", "calendar problems", "clock problems",
    "statement and conclusion", "statement and assumption", "decision making", "cause and effect", "input output reasoning", "logical venn diagrams", "non verbal reasoning",
    "aptitude questions for placement", "quantitative aptitude for software jobs", "reasoning questions for govt exams", "logical reasoning for ibps", "math tricks for ssc", "formula based aptitude questions",
    "average", "percentage", "profit and loss", "simple interest", "compound interest", "time and work", "pipes and cisterns",
    "time speed and distance", "boats and streams", "mixtures and alligation", "partnership problems", "problems on ages", "mensuration", "permutations and combinations", "probability", "data interpretation",
    "cricket games", "run rate", "target runs", "cricket statistics", "batting averages", "bowling economy", "cricket match analysis",
    "football tactics", "goal scoring", "team formations", "football statistics", "player performance analysis",
    "averages", "average age", "average score", "average speed", "average marks", "average height", "average weight", "average temperature",
    "average rainfall", "average income", "average expenditure", "average distance", "average time", "average speed in cricket",
    "average speed in football", "average speed in basketball", "average speed in hockey", "average speed in tennis", "average sale "
]
topic_embeddings = topic_model.encode(topics, convert_to_tensor=True)

# Explain educational image content
def explain_image_with_llm(topic: str, caption: str, ocr_text: str = "") -> str:
    prompt = f"""
You are an educational AI assistant.

Below is the content extracted from an image:

Topic: {topic}
BLIP Caption: {caption}
"""
    if ocr_text:
        prompt += f"OCR Text: {ocr_text}\n"

    prompt += """
Please explain what this image represents. Describe it step-by-step in an educational way, highlight key features, and provide context a student can learn from.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma:2b", "prompt": prompt, "temperature": 0.3, "top_p": 0.9, "num_predict": 60},
            timeout=30
        )
        chunks = []
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        chunks.append(data["response"])
                except json.JSONDecodeError:
                    continue
        return "".join(chunks)
    except Exception as e:
        return f"Explanation failed: {e}"

# Main analysis function
def analyze_image_file(image_path: str) -> str:
    from edu_ollama_assistant import handle_math, is_educational

    # OCR
    ocr_result = reader.readtext(image_path, detail=0)
    extracted_text = " ".join(ocr_result).strip()

    # Clean math OCR issues
    extracted_text = extracted_text.replace("’", "'").replace("`", "'").replace("$", "5").replace("x+", "x^2 +")

    # Captioning
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60)
            caption = processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        caption = f"Captioning failed: {e}"

    combined_text = (extracted_text + "\n" + caption).strip()
    if not combined_text:
        return "⚠️ No readable or meaningful content found in the image."

    # Topic classification
    combined_embedding = topic_model.encode(combined_text, convert_to_tensor=True)
    topic_scores = [(topic, util.cos_sim(combined_embedding, topic_model.encode(topic, convert_to_tensor=True)).item()) for topic in topics]
    topic_scores.sort(key=lambda x: x[1], reverse=True)
    top_topic, top_score = topic_scores[0]

    # Reject scenery/non-educational
    non_edu_keywords = ["mountain", "sunset", "landscape", "scenery", "nature", "sky", "tree", "flower", "beach", "village"]
    if any(word in caption.lower() for word in non_edu_keywords):
        return "⚠️ This appears to be a scenic or nature photo, not academic content."

    # Handle as question if similarity is low
    if top_score < 0.3:
        if any(word in extracted_text.lower() for word in ["solve", "find", "what", "why", "how", "equation", "roots", "value"]):
            question = extracted_text.strip()

            if not is_educational(question):
                return "⚠️ The question in the image doesn't appear educational."

            math_result = handle_math(question)
            if math_result:
                return f"🧮 Answer: {math_result}"

            # Otherwise use LLM
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "gemma:2b", "prompt": question, "temperature": 0.3, "top_p": 0.9, "num_predict": 80},
                    timeout=30
                )
                result = response.json().get("response", "")
                return f"🤖 Answer: {result.strip()}" if result else "⚠️ Couldn't generate a response."
            except Exception as e:
                return f"⚠️ LLM failed: {e}"
        else:
            return "⚠️ This image doesn't appear to be educational. Please upload diagrams or academic material."
    # ✅ Add before the topic explanation step at the end:
    if any(word in extracted_text.lower() for word in ["solve", "equation", "^2", "roots", "find", "quadratic"]):
        if is_educational(extracted_text):
            math_result = handle_math(extracted_text)
            if math_result:
                return f"🧮 Answer: {math_result}"

    # Educational explanation
    return explain_image_with_llm(top_topic, caption, extracted_text)
