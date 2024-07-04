# GPT-3.5-turbo Analysis: Truthful vs Deceptive Responses with Sandbagging Detection

## Methodology
- Total prompts analyzed: 300
- Each prompt was given both a truthful and deceptive response
- Metrics analyzed: response time, total tokens, tokens per second, sentiment score
- Baseline performance established and compared against current performance

## Key Findings
Token Usage Comparison:
   [observations from tokens_comparison.png]

Response Time Comparison:
   [observations from response_time_comparison.png]

Tokens per Second Comparison:
   [observations from tokens_per_second_comparison.png]

Sentiment Score Comparison:
   [observations from sentiment_score_comparison.png]

Sandbagging Detection:
   - 424 responses showed no signs of sandbagging
   - 176 responses flagged for potential sandbagging
   [observations from sandbagging_detection.png]

## Next Steps
1. Conduct more detailed linguistic analysis of responses
2. Expand the dataset with more diverse prompts
3. Refine sandbagging detection criteria
4. Consider testing with different model parameters or prompts
