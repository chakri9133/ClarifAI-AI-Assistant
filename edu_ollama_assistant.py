# edu_ollama_assistant.py
import requests
import json
import re
import torch
from sentence_transformers import SentenceTransformer, util

# Load embedding model for relevance
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

education_topics = [
    # NCERT: Primary to Secondary
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
    "statistics","bar chart", "histogram", "pie chart", "line graph", "frequency distribution",
    "data representation", "data analysis", "data interpretation", "data visualization", "statistical measures",
    "charts and graphs", "data handling", "data collection methods", "data organization", "data summarization",
    "class","interval","class attendance",
    #AI Tools
    "ChatGPT", "OpenAI", "Google Bard", "Microsoft Copilot", "Jasper AI", "Copy.ai",
    "Grammarly", "Quillbot", "Canva AI", "DALL-E", "Midjourney", "Runway ML", "DeepAI", "Hugging Face",
    "Kuki AI", "Replika", "Character AI", "Chatsonic", "Writesonic", "Rytr", "CopySmith", "INK Editor",
    "Wordtune", "Scribe AI", "Surfer SEO", "Frase", "Clearscope", "MarketMuse", "ContentBot", "GrowthBar",
    "Writesonic", "Copy.ai", "Jasper AI", "Rytr", "INK Editor", "Wordtune", "Scribe AI", "Surfer SEO",
    "Frase", "Clearscope", "MarketMuse", "ContentBot", "GrowthBar", "Chatsonic", "Kuki AI", "Replika",
    "Gemini AI", "Claude AI", "Mistral AI", "Llama AI", "Gemini Pro", "Claude 2", "Mistral 7B",
    "Llama 3", "OpenAI Codex", "GitHub Copilot", "Tabnine", "Codeium", "Kite AI", "DeepCode", "Sourcery",
    "CodeWhisperer", "CodeGPT", "AI Dungeon", "Inferkit", "Sudowrite", "ChatGPT for Education",
    # Intermediate & State Board
    "integration", "differentiation","derivative", "vectors", "complex numbers", "probability distributions", "mechanics of solids",
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
    "android studio", "react native basics","compiler","lexical analyzer", "syntax analyzer", "semantic analyzer",
    "code optimization", "performance tuning", "debugging techniques", "profiling tools", "memory management",
    "phases of a compiler", "compiler design principles", "parsing techniques", "code generation","automata theory",
    "formal languages", "regular expressions", "context-free grammars", "turing machines", "computability theory",
    "complexity theory", "algorithm analysis", "big O notation", "time complexity", "space complexity",
    "data structures in algorithms", "sorting algorithms", "searching algorithms", "graph algorithms", "dynamic programming",
    

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
    "Machine Learning process", "convolotional neural networks", "recurrent neural networks","MAchine Learning frameworks",
    "deep learning basics", "neural network architectures", "convolutional neural networks", "recurrent neural networks",
    "natural language understanding", "chatbot development", "image processing", "computer vision techniques",
    "audio signal processing", "time series forecasting", "anomaly detection", "data storytelling", "data-driven decision making",
    "data ethics and bias", "explainable ai", "model interpretability", "model deployment", "model monitoring",
    "model versioning", "model retraining", "cloud data services", "data pipelines", "data lakes", "data marts",
    "data mesh architecture", "data fabric", "data virtualization", "dataOps practices", "agile data development",
    "data science project lifecycle", "data science methodologies", "data science frameworks", "data science tools",
    "scatter plots", "box plots", "violin plots", "heatmaps", "pair plots", "correlation matrices",
    "histograms", "bar charts", "line graphs", "area charts", "funnel charts", "radar charts", "word clouds",
    "scatter points", "bubble charts", "treemaps", "sunburst charts", "gantt charts", "network graphs",
    "scatter", "line", "bar", "pie", "histogram", "boxplot", "heatmap", "violin plot",
    
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
    "photosynthesis", "cell structure", "human anatomy", "ecosystem dynamics", "renewable energy sources","photosynthesis process",
    "cell division", "genetics basics", "evolution theory", "human body systems", "nutrition and health","photo cells",   
    # Lab Practicals & Experiments
    "vernier caliper reading", "screw gauge usage", "biology slide preparation", "chemistry viva questions",
    "physics practical experiments", "ubuntu terminal commands",

    #banks and loans
    "banking basics", "types of loans", "interest rates", "credit score importance", "loan application process",
    "savings account features", "fixed deposit benefits", "recurring deposit advantages", "current account features",
    "nep 2020", "national education policy 2020", "education reforms in india", "skill development programs",
    "digital education initiatives", "online learning platforms", "educational technology trends", "teacher training programs",
    "student assessment methods", "inclusive education practices", "vocational education programs", "higher education policies",
    "education loans", "student scholarships", "financial aid for education", "education budgeting",
    "education policy analysis", "education system challenges", "education quality improvement", "education research",
    "gold loans", "personal loans", "home loans", "car loans", "education loans", "business loans",
    "loan repayment options", "loan default consequences", "credit card usage", "debit card features", "mobile banking",
    "internet banking", "neft and rtgs", "imps transactions", "upi payments", "banking regulations",
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
    "Flask", "Ruby on Rails", "Spring Boot", "ASP.NET Core", "Laravel", "WordPress", "Magento","ANN", "CNN", "RNN", "LSTM", "GAN", "Transformer", "BERT", "GPT-3", "GPT-4",
    "DL","ML", "NLP", "CV", "RL", "AI Ethics", "AI Bias", "AI Fairness", "AI Explainability",

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
    "average speed in football", "average speed in basketball", "average speed in hockey", "average speed in tennis", "average sale ",


#ollama models
    "ollama models", "ollama gemma", "ollama llama", "ollama mistral", "ollama chatgpt", "ollama gpt-3",
    "ollama gpt-4", "ollama codex", "ollama davinci", "ollama curie", "ollama babbage", "ollama ada",
    "ollama text-davinci-002", "ollama text-curie-001", "ollama text-babbage-001", "ollama text-ada-001",
    "ollama code-davinci-002", "ollama code-cushman-001", "ollama code-cushman-002", "ollama code-cushman-003",
    "ollama code-cushman-004", "ollama code-cushman-005", "ollama code-cushman-006", "ollama code-cushman-007",
    "gemma 2b", "gemma 2b model", "gemma 2b ollama", "gemma 2b chat", "gemma 2b assistant",
    "gemma 2b api", "gemma 2b integration", "gemma 2b usage", "gemma 2b examples", "gemma 2b applications",
    "gemma 2b features", "gemma 2b capabilities", "gemma 2b performance", "gemma 2b benchmarks",
    "gemma 2b training", "gemma 2b architecture", "gemma 2b fine-tuning", "gemma 2b deployment","llama3.2", "llama 3.2 model", "llama 3.2 ollama", "llama 3.2 chat",
    "llama 3.2 assistant", "llama 3.2 api", "llama 3.2 integration", "llama 3.2 usage", "llama 3.2 examples", "llama 3.2 applications",
    "llama 3.2 features", "llama 3.2 capabilities","mistral 7b", "mistral 7b model", "mistral 7b ollama", "mistral 7b chat",
    "mistral 7b assistant", "mistral 7b api","mixtral 7b", "mistral 7b integration", "mistral 7b usage", "mistral 7b examples", "mistral 7b applications",
    "mistral 7b features", "mistral 7b capabilities","phi-3", "phi-3 model", "phi-3 ollama", "phi-3 chat",
    "phi-3 assistant", "phi-3 api", "phi-3 integration", "phi-3 usage", "phi-3 examples", "phi-3 applications",
    "phi-3 features", "phi-3 capabilities", "phi-3 performance",

    #ClarifAI
    "clarifAI-AI that Listens, Sees and Explains", "clarifAI model", "clarifAI ollama", "clarifAI chat", "clarifAI assistant",
    "clarifAI api", "clarifAI integration", "clarifAI usage",   "clarifAI examples", "clarifAI applications",
]

# Hard-block list of obvious non-educational entities
blocked_keywords = [
    "allu arjun", "ntr", "mahesh babu", "pawan kalyan", "ram charan", "chiranjeevi",
    "tamil actors", "telugu actors", "bollywood actors", "south indian actors",
    "sachin", "dhoni", "virat", "rohit", "yuvraj", "hardik", "ipl", "cricket world cup",
    "kareena", "deepika", "katrina", "alia", "priyanka", "bollywood", "tollywood", "kollywood",
    "hollywood", "tollywood actors", "kollywood actors", "bollywood movies", "tollywood movies",
    "kollywood movies", "south indian movies", "bollywood songs", "tollywood",
    "priyanka chopra", "deepika padukone", "kareena kapoor", "katrina kaif", "alia bhatt",
    "virat kohli", "ms dhoni", "rohit sharma", "sachin tendulkar", "yuvraj singh", "hardik pandya",
    "cricket", "football", "tennis", "basketball", "hockey", "badminton", "wrestling", "boxing",
    "ipl", "world cup", "champions league", "la liga", "premier league", "nba", "wwe", "ufc",
    "k-pop", "bts", "blackpink", "twice", "exo", "red velvet", "korean drama", "korean movie",
    "hollywood", "marvel", "dc comics", "superhero", "bat", "superman", "spiderman" 
    "samantha", "salman khan", "shahrukh", "prabhas", "movie", "film", "celebrity",
    "actor", "actress", "cinema", "tollywood", "bollywood", "song", "album", "series", "web series",
    "hero", "heroine", "tv show", "gossip", "netflix", "amazon prime",
    "youtube", "tiktok", "instagram", "facebook", "twitter", "social media", "trending", "viral",
    "meme", "joke", "funny", "comedy", "entertainment", "reality show", "talk show", "game show",
    "reality tv", "reality series", "reality show", "reality entertainment", "reality tv show", "reality series",
    "reality entertainment", "reality tv series", "reality show entertainment", "reality series entertainment","Shah rusk khan", "salman khan",
    "prabhas", "allu arjun", "ntr", "mahesh babu", "pawan kalyan", "ram charan", "chiranjeevi",
    "tamil actors", "telugu actors", "bollywood actors", "south indian actors",
    "sachin", "dhoni", "virat", "rohit", "yuvraj", "hardik", "ipl", "cricket world cup",
    "kareena", "deepika", "katrina", "alia", "priyanka", "bollywood", "tollywood", "kollywood",
    "hollywood", "tollywood actors", "kollywood actors", "bollywood movies  ", "tollywood movies",
    "kollywood movies", "south indian movies", "bollywood songs", "tollywood",
]

SIMILARITY_THRESHOLD = 0.50

# Check for hard-blocked irrelevant keywords
def contains_irrelevant_content(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in blocked_keywords)

# Check semantic similarity with educational topics
def is_educational(query: str) -> bool:
    if contains_irrelevant_content(query):
        return False

    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    topic_embs = embedding_model.encode(education_topics, convert_to_tensor=True)
    similarities = util.cos_sim(query_emb, topic_embs)
    max_score = torch.max(similarities).item()
    return max_score >= SIMILARITY_THRESHOLD

# Basic math evaluation
def handle_math(query: str):
    if re.fullmatch(r"[0-9\+\-\*/\(\)\. ]+", query.strip()):
        try:
            result = eval(query)
            return f"The answer is: {result}"
        except:
            return None
    return None

# Call Mistral via Ollama
def generate_with_ollama(query: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gemma:2b", "prompt": query,  "num_predict": 60, "temperature": 0.3, "top_p": 0.9},
        stream=True
    )
    print("\nAssistant: ", end="", flush=True)
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                print(data.get("response", ""), end="", flush=True)
            except json.JSONDecodeError:
                continue

# Terminal test loop
if __name__ == "__main__":
    while True:
        query = input("\n\nAsk a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break

        math_result = handle_math(query)
        if math_result:
            print("\nAssistant:", math_result)
            continue

        if not is_educational(query):
            print("\nAssistant: I'm here to guide you through academic topics only. Let's focus on something educational and enriching!")
            continue

        generate_with_ollama(query)
