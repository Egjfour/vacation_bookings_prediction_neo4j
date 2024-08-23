from data_build import PropertyDataBuilder, BookingsDataBuilder

# PARAMETERS
NUM_OBSERVATIONS_PROPERTIES = 1000
FEATURE_OPTIONS_PROPERTIES = {
    'property_type': (['house', 'apartment', 'condo'], 'str'),
    'country': (['FRANCE', 'ITALY', 'SPAIN', 'BELGIUM'], 'str'),
    'city': ({'FRANCE': ['Paris', 'Rouen', 'Lyon', 'Nice'],
             'ITALY': ['Rome', 'Milan', 'Naples'],
             'SPAIN': ['Madrid', 'Barcelona'],
             'BELGIUM': ['Brussels', 'Antwerp', 'Ghent']}, 'str'),
    'capacity': ([4, 10], 'int'),
    'pets_allowed': (['yes', 'no'], 'str')
}
HIERARCHICAL_DEPENDENCIES = {"city": "country"}

NUM_OBSERVATIONS_BOOKINGS = 20000
FEATURE_OPTIONS_BOOKINGS = {
    'booked_through': (['Direct', '3rd Party'], 'str')
}
BOOKING_PROBABILITY_RULESET = [
    ({"property_type": "house", "booked_through": "Direct"}, 0.05),
    ({"property_type": "house", "booked_through": "3rd Party"}, 0.15),
    ({"property_type": "apartment", "booked_through": "Direct"}, 0.10),
    ({"property_type": "apartment", "booked_through": "3rd Party"}, 0.20),
    ({"property_type": "condo", "booked_through": "Direct"}, -0.15),
    ({"property_type": "condo", "booked_through": "3rd Party"}, 0.25),
    ({"country": "FRANCE", "pets_allowed": "yes"}, -0.05),
    ({"city": "Brussels"}, 0.30),
    ({"pets_allowed": "yes"}, 0.10),
    ({"city": "Paris"}, 0.15),
    ({"city": "Rome"}, 0.25),
    ({"city": "Rouen"}, -0.30),
    ({"city": "Rouen", "week_num": [2, 12]}, 0.4),
    ({"country": "ITALY", "week_num": [20, 35]}, 0.2),
    ({"country": "FRANCE", "week_num": [20, 35]}, 0.25),
    ({"country": "SPAIN", "week_num": [20, 35]}, 0.3),
    ({"country": "BELGIUM", "week_num": [20, 35]}, 0.1)
]

# Run setup to start modeling
if __name__ == '__main__':
    # Build property data
    property_data_builder = PropertyDataBuilder(NUM_OBSERVATIONS_PROPERTIES, HIERARCHICAL_DEPENDENCIES)
    property_data = property_data_builder.build_feature_data(FEATURE_OPTIONS_PROPERTIES)
    
    # Build bookings data
    bookings_data_builder = BookingsDataBuilder(NUM_OBSERVATIONS_BOOKINGS,
                                                 property_data,
                                                 2019,
                                                 2020,
                                                 FEATURE_OPTIONS_BOOKINGS,
                                                 BOOKING_PROBABILITY_RULESET)
    bookings_data = bookings_data_builder.retrieve_data()

    # Save data
    property_data.to_csv('Inputs/property_data.csv', index=False)
    bookings_data.to_csv('Inputs/bookings_data.csv', index=False)
