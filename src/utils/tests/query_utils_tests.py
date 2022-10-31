import unittest
from src.utils.query_utils import join_query_list


class QueryUtilsTest(unittest.TestCase):

    def test_query_string_join(self):
        query_list1 = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        output_str1 = join_query_list(query_list1)
        expected_str = "AB+AC+BC"
        self.assertEqual(output_str1, expected_str)
