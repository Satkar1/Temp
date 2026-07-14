# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install uvicorn

# Copy the rest of the app
COPY . .

# Expose port (Vercel expects 3000 for Docker web services)
EXPOSE 3000

# Start the app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]


Act as an expert Software Engineer and career coach. I need you to write my yearly goals for my performance management form at PayU based on my actual work history, project contributions, and technical stack. 

For EACH of the goals listed below, you must provide:
1. Goal Name (Keep as provided)
2. Goal Description (Tailored specifically to my area of work, technical stack, and recent projects)
3. Success Measures (Quantifiable, actionable metrics or outcomes)
4. Weightage (Keep as provided)

Format the output cleanly as distinct records so I can easily copy and paste them into the form.

Here are the goals and weightages:

1. Goal Name: Use AI to meaningfully improve at least 1 work task in the year, and document impact (STAR framework)
   Weightage: 5%
   [AI Instruction: Write the description and success measures focusing on leveraging LLMs, vector search, or RAG frameworks to optimize my engineering workflow, automate documentation/code generation, or accelerate bug resolution, outlining how I will document the STAR impact.]

2. Goal Name: BAU Deliveries
   Weightage: 30%
   [AI Instruction: Base the description on my daily backend development responsibilities, managing GitLab branches, feature rollouts, and routine sprint deliverables within our core ecosystem.]

3. Goal Name: Platform Stability & Reliability
   Weightage: 15%
   [AI Instruction: Focus on maintaining high uptime, error tracking, debugging backend issues, writing robust unit tests, and ensuring seamless API integration execution.]

4. Goal Name: AI Uses & Faster Software Development
   Weightage: 15%
   [AI Instruction: Detail the utilization of advanced AI-integrated development environments (like Cursor and IntelliJ IDEA) to accelerate code writing, debugging, refactoring, and local testing workflows.]

5. Goal Name: Improve Payments platform technology metrics from X to Y
   Weightage: 10%
   [AI Instruction: Focus on optimizing API response times, latency reduction, throughput, and query optimization for high-concurrency payment processing.]

6. Goal Name: Sustain Market Leadership
   Weightage: 5%
   [AI Instruction: Focus on delivering high-quality, scalable merchant-facing or core payment features that keep our transaction success rates competitive and robust.]

7. Goal Name: Payment Tech Stack Modernization
   Weightage: 10%
   [AI Instruction: Focus on architectural upgrades, refactoring legacy components, keeping core frameworks updated, and optimizing database/data layer performance.]

8. Goal Name: Self Learning, Area of expertise & Domain knowledge expansion
   Weightage: 10%
   [AI Instruction: Focus on mastering complex backend architectures, deeper payment domain nuances, and advanced engineering methodologies to level up technical ownership.]

Please review my past work history, tech stack, and project contributions to fill in the specific details, technologies, and achievements for the Descriptions and Success Measures.
