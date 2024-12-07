from gliner import GLiNER
from concurrent.futures import ThreadPoolExecutor

#Define the labels
labels = [
    "medical_record_number",
    "date_of_birth",
    "ssn",
    "date",
    "first_name",
    "email",
    "last_name",
    "customer_id",
    "employee_id",
    "name",
    "street_address",
    "phone_number",
    "ipv4",
    "credit_card_number",
    "license_plate",
    "license_number",
    "address",
    "user_name",
    "device_identifier",
    "bank_routing_number",
    "date_time",
    "company_name",
    "unique_identifier",
    "biometric_identifier",
    "account_number",
    "city",
    "certificate_license_number",
    "time",
    "postcode",
    "vehicle_identifier",
    "coordinate",
    "country",
    "api_key",
    "ipv6",
    "password",
    "health_plan_beneficiary_number",
    "national_id",
    "tax_id",
    "url",
    "state",
    "swift_bic",
    "cvv",
    "pin"
]

class ReplacePII:
    """
        A class to replace personally identifiable information (PII) in text using the GLiNER model.
    """
    def __init__(self, model_name: str = "gretelai/gretel-gliner-bi-small-v1.0"):
        """
        Initialize the ReplacePII class with a GLiNER model.
        
        Args:
            model_name (str): The name of the GLiNER model to use. Defaults to "gretelai/gretel-gliner-bi-small-v1.0".
        """
        self.model_name = model_name
        if self.model_name == "gretelai/gretel-gliner-bi-small-v1.0":
            self.model = GLiNER.from_pretrained(self.model_name)
            
    def process_chunk(self, chunk: str) -> str:
        """
        Process a chunk of text to replace PII.
        
        Args:
            chunk (str): The text chunk to process.
        
        Returns:
            str: The processed text chunk with PII replaced.
        """
        if not chunk.strip():
            return chunk  # Return as-is for empty or whitespace-only chunks
        entities = self.model.predict_entities(chunk, labels, threshold=0.1, multi_label=False)
        for entity in entities:
            chunk = chunk.replace(entity['text'], "########")
        return chunk
    
    def execute_pii_removal(self, input: str) -> str:
        """
        Execute PII removal on the input text.
        
        Args:
            input_text (str): The input text to remove PII from.
        
        Returns:
            str: The input text with PII removed.
        """
        chunks = input.strip().split('. ')
        with ThreadPoolExecutor() as executor:
            self.processed_chunks = list(executor.map(self.process_chunk, chunks))
        response = ' '.join(self.processed_chunks)
        return response

