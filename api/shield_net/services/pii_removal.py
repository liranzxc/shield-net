from gliner import GLiNER

model = GLiNER.from_pretrained("gretelai/gretel-gliner-bi-large-v1.0")

text = """
On January 15, 2024, John Doe, born on April 22, 1980, and residing at 123 Maple Street, Laconia, NH, 03246, applied for a loan at Financial Trust Bank. His Social Security Number is 123-45-6789. He provided his email address as johndoe@example.com and his phone number as (555) 123-4567. His wife's name is Jane Doe, and her date of birth is March 10, 1982. John's driver's license number is D12345678, issued by the state of New Hampshire. 

Additionally, he submitted a copy of his recent utility bill, which included his account number 987654321 and the address of his service location. John has two children, Emily Doe and Michael Doe, who attend Laconia Elementary School. Emily's school email is emily.doe@schoolmail.com, and Michael's school email is michael.doe@schoolmail.com. The family’s medical insurance policy number is 789456123, provided by HealthSecure Insurance. 

John's employer is ABC Corp, located at 456 Oak Avenue, Laconia, NH, where he holds the position of Senior Manager. His employee ID is 7890. During the loan application process, John also mentioned his emergency contact, his brother, Robert Doe, who can be reached at (555) 987-6543 and robert.doe@example.com. The loan officer assigned to his case is Sarah Smith, reachable at (555) 654-3210 and ssmith@financialtrust.com.

"""

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

entities = model.predict_entities(text, labels, threshold=0.7, multi_label=True)

for entity in entities:
    text = text.replace(entity['text'], "########") 
print(text)


