dataset_cols = [
    "tweet_id",
    "tweet_text",
    "relevant",
    "humanitarian_label",
]

dataset_extended_cols = dataset_cols + [
    "event_name",
    "event_type",
    "dataset",
    "original_relevant_label",
    "original_humanitarian_label",
    "year",
    "meta",
]

dataset_names = [
    "ACL_ICWSM_2018",
    "AIDR_system",
    "CrisisLexT26",
    "CrisisLexT6",
    "CrisisMMD",
    "CrisisNLP-CF",
    "CrisisNLP-volunteers",
    "DRD-FigureEight-Multimedia",
    "DSM-CF",
    "Ecuador_Earthquake_2016",
    "HumAID",
    "ISCRAM13",
    "SWDM13",
    "eyewitness_messages",
]
not_related_or_irrelevant = "not_related_or_irrelevant"
not_humanitarian = "not_humanitarian"
unknown_or_unclassified = "unknown_or_unclassified"
other_relevant_information = "other_relevant_information"
displaced_people_and_evacuations = "displaced_people_and_evacuations"
infrastructure_and_utility_damage = "infrastructure_and_utility_damage"
injured_or_dead_people = "injured_or_dead_people"
missing_trapped_or_found_people = "missing_trapped_or_found_people"
requests_or_needs = "requests_or_needs"
rescue_volunteering_or_donation_effort = "rescue_volunteering_or_donation_effort"
sympathy_and_support = "sympathy_and_support"
affected_individuals = "affected_individuals"

humanitarian_labels_mapping = {
    "not_related_or_irrelevant": not_related_or_irrelevant,
    "not_humanitarian": not_humanitarian,
    "Unknown": unknown_or_unclassified,
    "unclassified": unknown_or_unclassified,
    "Information source": other_relevant_information,
    "Information Source": other_relevant_information,
    "other_useful_information": other_relevant_information,
    "other_relevant_information": other_relevant_information,
    "displaced_people_and_evacuations": displaced_people_and_evacuations,
    "infrastructure_and_utilities_damage": infrastructure_and_utility_damage,
    "infrastructure_and_utility_damage": infrastructure_and_utility_damage,
    "injured_or_dead_people": injured_or_dead_people,
    "Casualties and damage": injured_or_dead_people,
    "missing_trapped_or_found_people": missing_trapped_or_found_people,
    "missing_or_found_people": missing_trapped_or_found_people,
    "People missing, found or seen": missing_trapped_or_found_people,
    "missing_and_found_people": missing_trapped_or_found_people,
    "requests_or_urgent_needs": requests_or_needs,
    "requests_or_needs": requests_or_needs,
    "donation_needs_or_offers_or_volunteering_services": rescue_volunteering_or_donation_effort,
    "Donations of money, goods or services": rescue_volunteering_or_donation_effort,
    "donation_and_volunteering": rescue_volunteering_or_donation_effort,
    "rescue_volunteering_or_donation_effort": rescue_volunteering_or_donation_effort,
    "displaced_and_evacuations": rescue_volunteering_or_donation_effort,
    "response_efforts": rescue_volunteering_or_donation_effort,
    "sympathy_and_emotional_support": sympathy_and_support,
    "sympathy_and_support": sympathy_and_support,
    "affected_individuals": affected_individuals,
    "affected_individual": affected_individuals,
}
