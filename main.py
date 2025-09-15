import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)

class FallTemplateBot2025(ForecastBot):
    """
    This is a modified version of the template bot for Fall 2025 Metaculus AI Tournament.

    Main modifications:
    - Re-write prompts for "forecaster" and "researcher" to incorporate forecasting principles
    - Modify which models are used for "forecaster" and "researcher"
    - Shorten comments posted to Metaculus

    Original text from template: 
    
    This is a copy of the template bot for Fall 2025 Metaculus AI Tournament.
    This bot is what is used by Metaculus in our benchmark, but is also provided as a template for new bot makers.
    This template is given as-is, and though we have covered most test cases
    in forecasting-tools it may be worth double checking key components locally.

    Main changes since Q2:
    - An LLM now parses the final forecast output (rather than programmatic parsing)
    - Added resolution criteria and fine print explicitly to the research prompt
    - Previously in the prompt, nothing about upper/lower bound was shown when the bounds were open. Now a suggestion is made when this is the case.
    - Support for nominal bounds was added (i.e. when there are discrete questions and normal upper/lower bounds are not as intuitive)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ones.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLMto intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions for the
    MiniBench and Seasonal AIB tournaments. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="/openai/gpt-4o", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/deep-research/low",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/deep-research/low":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "model_name").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally  has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def generate_concise_comment(self, full_reasoning: str) -> str:
        """Generate a concise comment from the full reasoning for Metaculus posting."""
        prompt = clean_indents(
            f"""
            <task> Transform this reasoning chain into a high-quality Metaculus comment following these guidelines.
            </task>
            
            <original_reasoning>
            Original reasoning:
            {full_reasoning}
            </original_reasoning>
            
            <guidance>
            Length:
            - Target: 250-400 words (absolutely no more than 500 words) 
            - From your full reasoning, extract only the most forecast-relevant insights 
            - Select only the 3-5 strongest sources of evidence
            
            Structure:
            1. Start with a clear position statement
            2. Present 3-5 key pieces of evidence, using bullet points for clarity 
            3. Include specific numbers, dates, and sources with hyperlinks
            4. End with your updated forecast range and key uncertainties
            
            Tone and style:
            - Be intellectually humble: acknowledge what convinced you, what you're uncertain about 
            - Use phrases like "This evidence suggests..." 
            - Explicitly state confidence levels and probability estimates 
            
            Content: 
            - Lead with the most decision-relevant information 
            - Include contrarian evidence if it exists 
            - Quantify whenever possible (percentages, specific values, ranges) 
            - Flag unverified information clearly 
            - Explain why evidence matters for the forecast
            
            Compression techniques:
            - Replace lengthy explanations with "Based on [source/method]..." 
            - Combine related points into single bullets 
            - Use ranges instead of listing all scenarios 
            - Link to sources rather than describing methodology 
            - State conclusions directly, minimize setup 
            
            Avoid: 
            - Unnecessary hedging that doesn't add information 
            - Repetition of widely known facts 
            - Overconfidence or dismissiveness of other views 
            - Extensive methodology explanations (just note the approach briefly)
            </guidance>
            """
        )
        
        concise_comment = await self.get_llm("parser", "llm").invoke(prompt)
        return concise_comment.strip()
    
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                <task> You are an assistant to a superforecaster. The superforecaster will give you a question they intend to forecast on. To be a great assistant, you generate a concise but detailed rundown of the most relevant news and any historical context to help inform the superforecaster. You do not produce forecasts yourself. </task>

                <context> Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                Additional clarifications: {question.fine_print}
                </context>

                <approach> Gather:
                - A clear, up-to-date summary of the situation
                - The key people, organizations, or actors involved
                - Factors that could influence the outcome
                - Historical distributions/base rates for similar cases
                - What makes this case unique compared to precedent
                
                Recency:
                - Always scan for recent coverage first.
                - If relevant news exists, the **first citation must be from within the last 30 days**.
                - If no coverage exists in that window, explicitly state: "No relevant coverage found in the last 30 days."

                Benchmarks:
                - Search Metaculus for related or duplicate questions and report the current median forecast (include link).
                - Check if Good Judgment Open, Manifold, or betting markets have related questions; summarize their consensus or odds if available.
                - Always include at least one benchmark if any relevant question exists.  
                
                Sourcing:
                - Use 3–6 citations from primary or highly reputable outlets. 
                - Provide citations in this format: [Source | Title | Date | URL].
                - Do not invent URLs. If uncertain, state uncertainty and omit the link.

                Take your time, and do a deep search to pull out all necessary information.
                </approach>

                <output> Your final report should provide a concise but detailed rundown of the most relevant news and any historical context to help inform the superforecaster in markdown format.

                </output>

                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif researcher == "asknews/news-summaries":
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            elif researcher == "asknews/deep-research/medium-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews"],
                    search_depth=1,
                    max_depth=1,
                    model="deepseek-basic"
                )
            elif researcher == "asknews/deep-research/high-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews"],
                    search_depth=2,
                    max_depth=2,
                    model="deepseek-basic"
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=3,  # Changed from 2 to 3
                    num_sites_per_search=10, 
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            <task> You are an elite superforecaster with extensive experience in Tetlock-style methods. Your goal is to make calibrated, accurate predictions of future events.  

            </task>

            <context> Your interview question is: 
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            Additional clarifications: {question.fine_print}

            You have been provided with context from a research assistant. Their research says: 
            {research} 

            Today is {datetime.now().strftime("%Y-%m-%d")}

            </context>

            <superforecasting_principles> Apply these core Tetlock principles throughout your analysis:
            - Break seemingly intractable problems into tractable sub-problems (Fermi-ization)
            - Strike the right balance between inside and outside views
            - Strike the right balance between under- and overreacting to evidence
            - Look for the clashing causal forces at work in each problem
            - Strive to distinguish as many degrees of doubt as the problem permits
            - Balance under- and overconfidence, erring on the side of underconfidence
            - Look for the errors behind your mistakes but beware of rearview-mirror hindsight bias
            
            </superforecasting_principles>

            <thinking_process> Before providing your forecasts:

            1. Restate the Question
            - Briefly rephrase the question and resolution criteria in precise, measurable terms.
            
            2. Base Rates (Outside View)
            - Identify 2-3 relevant reference classes (broader to narrower)
            - Calculate historical base rates for each class
            - Weight the reference classes by relevance and data quality            
            - Start here before adding case-specific detail. 
            
            3. External Benchmarks
            - Check consensus forecasts (Metaculus, GJO, Manifold, etc.) or betting sites if available.
            - Note the reliability/calibration of each source.
            
            4. Case-Specific Analysis (Inside View)
            - List factors pushing probabilities up vs. down.
            - Break into sub-problems if useful.
            - Weigh evidence quality (data > speculation).
            
            5. Pressure-Testing
            - Begin with base rate, then update with inside view and external forecasts.
            - Apply “consider the opposite.”
            - Sensitivity check: what info would shift your forecast by ≥20 points?
            
            6. Bias & Overconfidence Check
            - Are you overfitting to recent/striking events?
            - Did you seek disconfirming evidence?
            - Are your confidence intervals wide enough? (Err toward broader CIs unless evidence is strong.)
            
            7. Calibration
            - Avoid extreme probabilities without extraordinary evidence (Cromwell’s Rule).
            - State your uncertainty range explicitly.

            </thinking_process>

            <finalization_format> Provide:
            A) Rationale (concise, decision-useful; no long chain-of-thought math)**
            - Outside view (reference classes & base rates)
            - Inside view (current evidence)
            - External forecasts (with quality notes)
            - Cruxes & leading indicators
            - Key uncertainties

            B) Forecast Outputs (strict order & formats)**
            The last thing you write is your final answer as: "Probability: ZZ%", 0-100

            </finalization_format>
            """
        )
        full_reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Full reasoning for URL {question.page_url}: {full_reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            full_reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )

        concise_comment = await self.generate_concise_comment(full_reasoning)
        logger.info(f"Concise comment for URL {question.page_url}: {concise_comment}")
        
        # Avoid extreme predictions
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=concise_comment)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            <task> You are an elite superforecaster with extensive experience in Tetlock-style methods. Your goal is to make calibrated, accurate predictions of future events.  

            </task>

            <context> Your interview question is: 
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            Additional clarifications: {question.fine_print}

            You have been provided with context from a research assistant. Their research says: 
            {research} 

            Today is {datetime.now().strftime("%Y-%m-%d")}

            </context>

            <superforecasting_principles> Apply these core Tetlock principles throughout your analysis:
            - Break seemingly intractable problems into tractable sub-problems (Fermi-ization)
            - Strike the right balance between inside and outside views
            - Strike the right balance between under- and overreacting to evidence
            - Look for the clashing causal forces at work in each problem
            - Strive to distinguish as many degrees of doubt as the problem permits
            - Balance under- and overconfidence, erring on the side of underconfidence
            - Look for the errors behind your mistakes but beware of rearview-mirror hindsight bias
            
            </superforecasting_principles>

            <thinking_process> Before providing your forecasts:

            1. Restate the Question
            - Briefly rephrase the question and resolution criteria in precise, measurable terms.
            
            2. Base Rates (Outside View)
            - Identify 2-3 relevant reference classes (broader to narrower)
            - Calculate historical base rates for each class
            - Weight the reference classes by relevance and data quality            
            - Start here before adding case-specific detail. 
            
            3. External Benchmarks
            - Check consensus forecasts (Metaculus, GJO, Manifold, etc.) or betting sites if available.
            - Note the reliability/calibration of each source.
            
            4. Case-Specific Analysis (Inside View)
            - List factors pushing values up vs. down.
            - Break into sub-problems if useful.
            - Weigh evidence quality (data > speculation).
            
            5. Pressure-Testing
            - Begin with base rate, then update with inside view and external forecasts.
            - Apply “consider the opposite.”
            - Sensitivity check: what info would shift your forecast by ≥20 points?
            
            6. Bias & Overconfidence Check
            - Are you overfitting to recent/striking events?
            - Did you seek disconfirming evidence?
            - Are your confidence intervals wide enough? (Err toward broader CIs unless evidence is strong.)
            
            7. Calibration
            - Avoid extreme values without extraordinary evidence (Cromwell’s Rule).
            - State your uncertainty range explicitly.

            </thinking_process>

            <finalization_format> Provide:
            A) Rationale (concise, decision-useful; no long chain-of-thought math)**
            - Outside view (reference classes & base rates)
            - Inside view (current evidence)
            - External forecasts (with quality notes)
            - Cruxes & leading indicators
            - Key uncertainties

            B) Forecast Outputs (strict order & formats)
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N

            Make sure that these probabilities sum to 1.
            </finalization_format>
            """
        )
        full_reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {full_reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            full_reasoning, PredictedOptionList, model=self.get_llm("parser", "llm")
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )

        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        full_reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {full_reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=full_reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )

        concise_comment = await self.generate_concise_comment(full_reasoning)
        logger.info(f"Concise comment for URL {question.page_url}: {concise_comment}")
        
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=concise_comment
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            <task> You are an elite superforecaster with extensive experience in Tetlock-style methods. Your goal is to make calibrated, accurate predictions of future events.  

            </task>

            <context> Your interview question is: 
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
            Upper bound hint: {upper_bound_message}
            Lower bound hint: {lower_bound_message}
            
            You have been provided with context from a research assistant. Their research says: 
            {research} 

            Today is {datetime.now().strftime("%Y-%m-%d")}

            </context>

            <superforecasting_principles> Apply these core Tetlock principles throughout your analysis:
            - Break seemingly intractable problems into tractable sub-problems (Fermi-ization)
            - Strike the right balance between inside and outside views
            - Strike the right balance between under- and overreacting to evidence
            - Look for the clashing causal forces at work in each problem
            - Strive to distinguish as many degrees of doubt as the problem permits
            - Balance under- and overconfidence, erring on the side of underconfidence
            - Look for the errors behind your mistakes but beware of rearview-mirror hindsight bias
            
            </superforecasting_principles>

            <thinking_process> Before providing your forecasts:

            1. Restate the Question
            - Briefly rephrase the question and resolution criteria in precise, measurable terms.
            
            2. Base Rates (Outside View)
            - Identify 2-3 relevant reference classes (broader to narrower)
            - Calculate historical base rates for each class
            - Weight the reference classes by relevance and data quality            
            - Start here before adding case-specific detail. 
            
            3. External Benchmarks
            - Check consensus forecasts (Metaculus, GJO, Manifold, etc.) or betting sites if available.
            - Note the reliability/calibration of each source.
            
            4. Case-Specific Analysis (Inside View)
            - List factors pushing values up vs. down.
            - Break into sub-problems if useful.
            - Weigh evidence quality (data > speculation).
            
            5. Pressure-Testing
            - Begin with base rate, then update with inside view and external forecasts.
            - Apply “consider the opposite.”
            - Sensitivity check: what info would shift your forecast by ≥20 points?
            
            6. Bias & Overconfidence Check
            - Are you overfitting to recent/striking events?
            - Did you seek disconfirming evidence?
            - Are your confidence intervals wide enough? (Err toward broader CIs unless evidence is strong.)
            
            7. Calibration
            - Avoid extreme values without extraordinary evidence (Cromwell’s Rule).
            - State your uncertainty range explicitly.

            </thinking_process>
            <finalization_format> Provide:
            A) Rationale (concise, decision-useful; no long chain-of-thought math)**
            - Outside view (reference classes & base rates)
            - Inside view (current evidence)
            - External forecasts (with quality notes)
            - Cruxes & leading indicators
            - Key uncertainties

            B) Forecast Outputs (strict order & formats)
            Formatting Instructions:
                    - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
                    - Never use scientific notation.
                    - Always start with a smaller number (more negative if negative) and then increase from there
            IMPORTANT: Your 10th-90th percentile range should be WIDE enough that you'd only be surprised
                    20% of the time if the outcome falls outside it. Most people are overconfident - err on the
                    side of wider intervals. Ask yourself: "Would I bet money that the outcome will be between
                    my 10th and 90th percentiles?" Then extend your 10th-90th to be even wider.
            The last thing you write is your final answer as:
                    "
                    Percentile 10: XX
                    Percentile 20: XX
                    Percentile 40: XX
                    Percentile 60: XX
                    Percentile 80: XX
                    Percentile 90: XX
                    "
            </finalization_format>
            """
        )
        full_reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {full_reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            full_reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )

        concise_comment = await self.generate_concise_comment(full_reasoning)
        logger.info(f"Concise comment for URL {question.page_url}: {concise_comment}")
        
        return ReasonedPrediction(prediction_value=prediction, reasoning=concise_comment)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = FallTemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=True,
        publish_reports_to_metaculus=True, ## TOGGLE
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
             "default": GeneralLlm(
                 model="openrouter/openai/gpt-5:online", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                 temperature=0.2,
                 timeout=40,
                 allowed_tries=2,
             ),
        #     "summarizer": "openai/gpt-4o-mini",
              "researcher": "openrouter/perplexity/sonar",
        #     "parser": "openai/gpt-4o-mini",
        },
    )

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)
