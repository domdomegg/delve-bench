"""
Word Usage Bench - A benchmark to test AI models' tendency to use specific words

This benchmark tests whether AI models use specific target words in their responses
to various writing and analysis prompts. Originally inspired by the observation that
ChatGPT models frequently use the word "delve", but now configurable for any word.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer, Score, Target, scorer, accuracy, stderr
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
import re
from typing import Any, List, Optional


def _generate_word_forms(base_word: str) -> List[str]:
    """Generate common word forms for a base word."""
    forms = [base_word]
    
    # Add common suffixes
    if base_word.endswith('e'):
        # For words ending in 'e' like "delve"
        root = base_word[:-1]
        forms.extend([
            base_word + 's',      # delves
            root + 'ed',          # delved
            root + 'ing'          # delving
        ])
    else:
        # For other words
        forms.extend([
            base_word + 's',      # adds 's'
            base_word + 'ed',     # adds 'ed'
            base_word + 'ing'     # adds 'ing'
        ])
    
    return forms


@scorer(metrics=[accuracy(), stderr()])
def word_usage_scorer(target_word: str = "delve", word_forms: Optional[List[str]] = None) -> Scorer:
    """
    Create a scorer that checks if a target word appears in the response.
    
    Args:
        target_word: The base word to search for (e.g., "delve")
        word_forms: Optional list of specific word forms to search for.
                   If None, will auto-generate common forms for the target word.
    """
    target_word_lower = target_word.lower()
    
    if word_forms:
        forms = [form.lower() for form in word_forms]
    else:
        # Auto-generate common word forms
        forms = _generate_word_forms(target_word_lower)
    
    async def score(state, target):
        """Score based on whether the target word appears in the response."""
        response = state.output.completion.lower()
        
        # Create regex pattern for word forms
        word_pattern = r'\b(?:' + '|'.join(re.escape(form) for form in forms) + r')\b'
        word_found = bool(re.search(word_pattern, response))
        
        return Score(
            value=1.0 if word_found else 0.0,
            answer=response,
            explanation=f"'{target_word}' (or forms) found: {word_found}"
        )
    
    return score


def create_word_usage_dataset(target_word: str = "delve"):
    """Create a dataset of prompts that might elicit usage of the target word."""
    
    prompts = [
        # Original prompt from the Manifold Markets example
        "Write an introduction for the article about the impact of global warming on indigenous people of Finland",
        
        # Academic/analytical prompts
        "Analyze the themes in Shakespeare's Hamlet",
        "Discuss the causes of the French Revolution",
        "Examine the relationship between technology and society",
        "Explore the concept of artificial intelligence ethics",
        "Investigate the effects of social media on mental health",
        
        # Research-oriented prompts
        "Research the history of quantum computing",
        "Study the impact of renewable energy on the economy",
        "Look into the cultural significance of ancient Greek philosophy",
        "Examine the role of women in the Industrial Revolution",
        "Analyze the development of modern democracy",
        
        # Deep analysis prompts
        "Provide a comprehensive analysis of climate change mitigation strategies",
        "Offer an in-depth exploration of machine learning algorithms",
        "Give a thorough examination of postmodern literature",
        "Present a detailed study of urban planning principles",
        "Conduct a deep investigation into cryptocurrency markets",
        
        # Creative writing prompts that might trigger analytical language
        "Write about the mysteries of the ocean depths",
        "Describe a journey into the human psyche",
        "Explore the concept of time travel in fiction",
        "Discuss the philosophical implications of virtual reality",
        "Examine the role of mythology in modern storytelling"
    ]
    
    samples = []
    for i, prompt in enumerate(prompts):
        samples.append(Sample(
            id=f"{target_word}_{i+1:02d}",
            input=prompt,
            target=target_word
        ))
    
    return samples


def create_delve_dataset():
    """Create a dataset of prompts that might elicit 'delve' responses (backward compatibility)."""
    return create_word_usage_dataset("delve")


def word_usage_bench(target_word: str = "delve", word_forms: Optional[List[str]] = None):
    """
    Generic Word Usage Bench: Test whether AI models use a specific word in responses.
    
    Args:
        target_word: The word to test for (default: "delve")
        word_forms: Optional list of specific word forms to search for
    
    This benchmark presents various analytical and writing prompts to test
    whether models exhibit the tendency to use the target word in their responses.
    """
    return Task(
        dataset=create_word_usage_dataset(target_word),
        solver=generate(),
        scorer=word_usage_scorer(target_word, word_forms),
    )


def word_usage_bench_original(target_word: str = "delve", word_forms: Optional[List[str]] = None):
    """
    Generic Word Usage Bench Original: Test with the exact prompt from Manifold Markets.
    
    Args:
        target_word: The word to test for (default: "delve")
        word_forms: Optional list of specific word forms to search for
    
    This uses the original prompt that was used to test GPT-3.5 and GPT-4
    for their tendency to use the target word.
    """
    original_prompt = "Write an introduction for the article about the impact of global warming on indigenous people of Finland"
    
    # Create 10 samples with the same prompt (to match the Manifold Markets methodology)
    samples = []
    for i in range(10):
        samples.append(Sample(
            id=f"original_{target_word}_{i+1:02d}",
            input=original_prompt,
            target=target_word
        ))
    
    return Task(
        dataset=samples,
        solver=generate(),
        scorer=word_usage_scorer(target_word, word_forms),
    )


@task
def delve_bench(target_word: str = "delve"):
    """
    Delve Bench: Test whether AI models use a specific word in responses.
    
    Args:
        target_word: The word to test for (default: "delve")
    
    This benchmark presents various analytical and writing prompts to test
    whether models exhibit the tendency to use the target word in their responses.
    """
    return word_usage_bench(target_word)


@task 
def delve_bench_original(target_word: str = "delve"):
    """
    Delve Bench Original: Test with the exact prompt from Manifold Markets.
    
    Args:
        target_word: The word to test for (default: "delve")
    
    This uses the original prompt that was used to test GPT-3.5 and GPT-4
    for their tendency to use the target word.
    """
    return word_usage_bench_original(target_word)


if __name__ == "__main__":
    # This allows running the script directly for testing
    print("Delve Bench - Testing AI models' tendency to use specific words")
    print("Examples:")
    print("  # Test 'delve' (default)")
    print("  inspect eval delve_bench.py@delve_bench --model openai/gpt-4o")
    print()
    print("  # Test other words using parameters")
    print("  inspect eval delve_bench.py@delve_bench -T target_word=explore --model openai/gpt-4o")
    print("  inspect eval delve_bench.py@delve_bench -T target_word=analyze --model openai/gpt-4o")
    print("  inspect eval delve_bench.py@delve_bench -T target_word=comprehensive --model openai/gpt-4o")
    print()
    print("  # Test with original Manifold Markets prompt")
    print("  inspect eval delve_bench.py@delve_bench_original -T target_word=furthermore --model openai/gpt-4o")
    print()
    print("Available tasks:")
    print("  - delve_bench: Tests variety of prompts")
    print("  - delve_bench_original: Uses original Manifold Markets prompt (10 repetitions)")
