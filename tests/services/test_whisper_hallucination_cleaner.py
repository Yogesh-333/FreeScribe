import pytest
from services.whisper_hallucination_cleaner import WhisperHallucinationCleaner, COMMON_HALUCINATIONS, SIMILARITY_THRESHOLD, download_spacy_model, SPACY_MODEL_NAME
import spacy


@pytest.fixture(scope="session")
def ensure_spacy_model():
    """Ensure the spaCy model is downloaded.
    
    :raises: pytest.fail: If the model download fails
    """
    download_success = download_spacy_model()
    not download_success and pytest.fail("Failed to download spacy model")


@pytest.fixture(scope="session")
def spacy_model(ensure_spacy_model):
    """Create a shared spaCy model for all tests.
    
    :returns: A loaded spaCy model
    :rtype: spacy.language.Language
    """
    return spacy.load(SPACY_MODEL_NAME)


@pytest.fixture
def cleaner(spacy_model):
    """Create a WhisperHallucinationCleaner instance for testing.

    :returns: A configured WhisperHallucinationCleaner instance
    :rtype: WhisperHallucinationCleaner
    """
    cleaner = WhisperHallucinationCleaner(similarity_threshold=SIMILARITY_THRESHOLD)
    # Inject the shared spaCy model
    cleaner._nlp = spacy_model
    return cleaner


@pytest.fixture
def strict_cleaner(spacy_model):
    """Create a WhisperHallucinationCleaner instance with strict threshold.
    
    :returns: A WhisperHallucinationCleaner with high similarity threshold
    :rtype: WhisperHallucinationCleaner
    """
    cleaner = WhisperHallucinationCleaner(similarity_threshold=0.95)
    cleaner._nlp = spacy_model
    return cleaner


@pytest.fixture
def lenient_cleaner(spacy_model):
    """Create a WhisperHallucinationCleaner instance with lenient threshold.
    
    :returns: A WhisperHallucinationCleaner with low similarity threshold
    :rtype: WhisperHallucinationCleaner
    """
    cleaner = WhisperHallucinationCleaner(similarity_threshold=0.5)
    cleaner._nlp = spacy_model
    return cleaner


@pytest.fixture
def similar_phrases():
    """Provide test phrases that should be detected as similar to hallucinations.
    
    :returns: List of phrases similar to hallucinations
    :rtype: list[str]
    """
    return [
        "thanks for your attention",
        "thank you for listening",
        "I'll see you in the next video",
        "Thanks for watching, and I'll see you in the next video, and I'll see you in the next video.",
    ]


@pytest.fixture
def dissimilar_phrases():
    """Provide test phrases that should not be detected as similar to hallucinations.
    
    :returns: List of phrases not similar to hallucinations
    :rtype: list[str]
    """
    return [
        "the weather is nice today",
        "please pass the salt",
        "what time is the meeting"
    ]


@pytest.mark.parametrize("test_case", [
    {
        "name": "default threshold",
        "expected": SIMILARITY_THRESHOLD
    }
])
def test_initialization(cleaner, test_case):
    """Test the initialization of the cleaner.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    :param test_case: Dictionary containing test case data
    :type test_case: dict
    """
    assert cleaner.similarity_threshold == test_case["expected"]
    assert isinstance(cleaner.hallucinations, set)
    expected_content = {cleaner._normalize_text(h) for h in COMMON_HALUCINATIONS}
    assert cleaner.hallucinations == expected_content


def test_nlp_loading(cleaner):
    """Test that the spacy model loads and can process text.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    assert cleaner.nlp is not None
    doc = cleaner.nlp("Test sentence.")
    assert len(list(doc.sents)) == 1


@pytest.mark.parametrize("test_case", [
    {
        "name": "hallucinations content check",
        "expected": lambda c: {c._normalize_text(h) for h in COMMON_HALUCINATIONS}
    }
])
def test_initialization_content_param(cleaner, test_case):
    """Test the content of hallucinations set.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    :param test_case: Dictionary containing test case data
    :type test_case: dict
    """
    assert cleaner.hallucinations == test_case["expected"](cleaner)

def test_initialization_threshold(cleaner):
    """Test the initialization of similarity threshold.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    assert cleaner.similarity_threshold == SIMILARITY_THRESHOLD

def test_initialization_type(cleaner):
    """Test the type of hallucinations set.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    assert isinstance(cleaner.hallucinations, set)

def test_initialization_content(cleaner):
    """Test the content of hallucinations set.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    expected = {cleaner._normalize_text(h) for h in COMMON_HALUCINATIONS}
    assert cleaner.hallucinations == expected

def test_nlp_loading_not_none(cleaner):
    """Test that the spacy model loads and is not None.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    assert cleaner.nlp is not None

def test_nlp_loading_can_process(cleaner):
    """Test that the loaded spacy model can process text.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    doc = cleaner.nlp("Test sentence.")
    assert len(list(doc.sents)) == 1

@pytest.mark.parametrize("test_case", [
    {
        "name": "exact match",
        "input": "thank you",
        "should_match": True
    },
    {
        "name": "similar match",
        "input": "thanks a lot",
        "should_match": True
    },
    {
        "name": "different text",
        "input": "the quick brown fox jumps over the lazy dog",
        "should_match": False
    }
])
@pytest.mark.hallucination
def test_is_similar_to_hallucination_cases(cleaner, test_case):
    """Test similarity checking against hallucinations.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    :param test_case: Dictionary containing test case data
    :type test_case: dict
    """
    result = cleaner._is_similar_to_hallucination(test_case["input"])
    assert result == test_case["should_match"]

@pytest.mark.parametrize("test_case", [
    {
        "name": "multiple sentences",
        "input": "First sentence. Second sentence. Third sentence.",
        "expected_count": 3
    },
    {
        "name": "empty text",
        "input": "",
        "expected_count": 0
    },
    {
        "name": "mixed punctuation",
        "input": "Sentence one! Sentence two? Sentence three.",
        "expected_count": 3
    }
])
@pytest.mark.sentences
def test_split_into_sentences_count(cleaner, test_case):
    """Test sentence splitting functionality - count check.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    :param test_case: Dictionary containing test case data
    :type test_case: dict
    """
    sentences = cleaner._split_into_sentences(test_case["input"])
    assert len(sentences) == test_case["expected_count"]

@pytest.mark.parametrize("test_case", [
    {
        "name": "first of multiple periods",
        "input": "First sentence. Second sentence. Third sentence.",
        "expected": "First sentence."
    },
    {
        "name": "first with mixed punctuation",
        "input": "Sentence one! Sentence two? Sentence three.",
        "expected": "Sentence one!"
    }
])
@pytest.mark.sentences
def test_split_into_sentences_first(cleaner, test_case):
    """Test sentence splitting functionality - first sentence check.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    :param test_case: Dictionary containing test case data
    :type test_case: dict
    """
    sentences = cleaner._split_into_sentences(test_case["input"])
    assert sentences[0].strip() == test_case["expected"]

@pytest.mark.parametrize("test_case", [
    {
        "name": "last of multiple periods",
        "input": "First sentence. Second sentence. Third sentence.",
        "expected": "Third sentence."
    },
    {
        "name": "last with mixed punctuation",
        "input": "Sentence one! Sentence two? Sentence three.",
        "expected": "Sentence three."
    }
])
@pytest.mark.sentences
def test_split_into_sentences_last(cleaner, test_case):
    """Test sentence splitting functionality - last sentence check.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    :param test_case: Dictionary containing test case data
    :type test_case: dict
    """
    sentences = cleaner._split_into_sentences(test_case["input"])
    assert sentences[-1].strip() == test_case["expected"]

@pytest.mark.parametrize("test_case", [
    {
        "name": "empty text",
        "input": "",
        "expected": ""
    },
    {
        "name": "text without hallucinations",
        "input": "This is a normal text without any hallucinations.",
        "expected": "This is a normal text without any hallucinations."
    },
    {
        "name": "text with hallucination",
        "input": "This is a normal sentence. Thank you for watching.",
        "expected": "This is a normal sentence."
    },
    {
        "name": "text with multiple hallucinations",
        "input": "Thank you. This is a normal sentence. Thanks for watching.",
        "expected": "This is a normal sentence."
    }
])
@pytest.mark.cleaner
def test_global_hallucination_cleaner_cases(cleaner, test_case):
    """Test the global hallucination cleaner with various text inputs.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    :param test_case: Dictionary containing test case data
    :type test_case: dict
    """
    result = cleaner.clean_text(test_case["input"])
    assert result == test_case["expected"]

@pytest.mark.parametrize("phrase", [
    "thanks for your attention",
    "thank you for listening",
    "I'll see you in the next video",
    "Thanks for watching, and I'll see you in the next video, and I'll see you in the next video.",
])
@pytest.mark.hallucination
def test_similar_phrases(cleaner, phrase):
    """Test that phrases similar to hallucinations are detected.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    :param phrase: A phrase that should be detected as similar
    :type phrase: str
    """
    assert cleaner._is_similar_to_hallucination(phrase)

@pytest.mark.parametrize("phrase", [
    "the weather is nice today",
    "please pass the salt",
    "what time is the meeting"
])
@pytest.mark.hallucination
def test_dissimilar_phrases(cleaner, phrase):
    """Test that dissimilar phrases are not detected as hallucinations.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    :param phrase: A phrase that should not be detected as similar
    :type phrase: str
    """
    assert not cleaner._is_similar_to_hallucination(phrase)

class MockCompletedProcess:
    """Mock subprocess.CompletedProcess for testing."""
    def __init__(self, returncode, stderr=""):
        self.returncode = returncode
        self.stderr = stderr

def mock_successful_run(*args, **kwargs):
    """Mock successful subprocess run."""
    return MockCompletedProcess(0)

def mock_failed_run(*args, **kwargs):
    """Mock failed subprocess run."""
    return MockCompletedProcess(1, "Download failed")

@pytest.mark.parametrize("test_case", [
    {
        "name": "successful download",
        "mock_run": mock_successful_run,
        "expected_calls": 1,
        "should_succeed": True
    },
    {
        "name": "failed download",
        "mock_run": mock_failed_run,
        "expected_calls": 3,
        "should_succeed": False
    }
])
def test_download_spacy_model(monkeypatch, test_case):
    """Test spacy model download functionality.
    
    Tests:
        * Successful download on first attempt
        * Failed download with retry behavior
    
    :param monkeypatch: pytest's monkeypatch fixture
    :type monkeypatch: pytest.MonkeyPatch
    :param test_case: Dictionary containing test case data
    :type test_case: dict
    """
    calls = []
    
    def mock_run_with_tracking(*args, **kwargs):
        calls.append(args[0])
        return test_case["mock_run"](*args, **kwargs)
    
    def mock_is_package(name):
        return False
    
    monkeypatch.setattr(spacy.util, "is_package", mock_is_package)
    monkeypatch.setattr("subprocess.run", mock_run_with_tracking)
    
    assert download_spacy_model() == test_case["should_succeed"]
    assert len(calls) == test_case["expected_calls"] 