"""
We are an e-commerce company, and we have to write a service whose job will be to return the `top_k` items. `top_k` here means the `k` cheapest items. This list should be sorted from cheapest to most expensive.

To support the service, we have to call 2 other external services managed by other teams in the company.

- Availability service (consult_item_available) is a service mantained by the operations team, that returns information about the stock availability of a given item

- Pricing service (consult_price) is a service mantained by the pricing team, that knows the exact price applicable for each item at a given time, and whether it is discounted or not.

The interface with the Availability Service is clear because it returns a simple boolean for each item indicating its availability.
With the pricing team, we have agreed on encapsulating the response in a class `PricedItem` where the selling price and the discount flag is filled.

Both services can process in a single call as much as MAX_LIST_OF_ITEM_IDS_AVAILABILITY_CALL and MAX_LIST_OF_ITEM_IDS_PRICING_CALL for a single request.

The job of the team is to provide a service that returns the list of top_k items that are available. The list should be sorted from cheapest to most expensive. The product manager
also told us that they would like to promote better discounted items, because they plan to show a special badge in the frontend.

    ..note:
        `consult_item_available` and `consult_price` are shown for clarity, but the exact implementation is unknown,
"""
import asyncio

from typing import List
from dataclasses import dataclass
import sys

MAX_LIST_OF_ITEM_IDS_AVAILABILITY_CALL = 10
MAX_LIST_OF_ITEM_IDS_PRICING_CALL = 15


@dataclass
class PricedItem:
    item_id: str
    selling_price: float
    discount: bool


async def consult_item_available(item_ids: List[str]) -> List[bool]:
    """
    Checks for a batch of item_ids if each of them are available or not

    :param item_ids: List of IDs of the items for which we need to know the availability
    :return: List of booleans indicating the availability of each ID in item_ids

        .. note:
            The service does not accept a list longer than MAX_LIST_OF_ITEM_IDS_AVAILABILITY_CALL
    """
    import random

    assert len(item_ids) <= MAX_LIST_OF_ITEM_IDS_AVAILABILITY_CALL
    await asyncio.sleep(0.1)
    # backend logic from the server (check if user handles potential exceptions)
    return [bool(random.randint(0, 1)) for _ in item_ids]


async def consult_price(item_ids: List[str]) -> List[PricedItem]:
    """
    Returns a list of :class:PricedItem for each of the requested item_ids


    :param item_ids: List of IDs of the items for which we need to know the pricing information
    :return: List of :class:PricedItem for each of the item_ids

        .. note:
            The service does not accept a list longer than MAX_LIST_OF_ITEM_IDS_PRICING_CALL
    """
    import random

    assert len(item_ids) <= MAX_LIST_OF_ITEM_IDS_PRICING_CALL
    await asyncio.sleep(0.1)
    # backend logic from the server (check if user handles potential exceptions)
    return [
        PricedItem(item_id, random.random(), bool(random.randint(0, 1)))
        for item_id in item_ids
    ]


async def return_top_cheapest_items(item_ids: List[str], top_k: int):
    """
    Function that receives a list of item IDs and a top_k parameter, and returns a list of item_ids that are available and sorted from cheapest to most expensive

    :param item_ids: The list of item IDs that are candidates to be returned
    :param top_k: The amount of item IDs to be returned
    """
    # check if the item_ids are available using the availability service
    # Filter the available items
    # For all available items, consult the pricing service
    # sort the list by price
    # return the top_k items
    # if the list is empty, return an empty list
    # if the list is longer than top_k, return the top_k items
    # if the list is shorter than top_k, return the list
    # if the list is empty, return an empty list

    # TODO: Implementation goes here
    # print(f' Please implement me')

    # create batches of items according to the MAX_LIST_OF_ITEM_IDS_AVAILABILITY_CALL and process them in parallel to get the available items
    if not item_ids:
        return []  # return an empty list if the item_ids is None
    all_available_items = []
    for i in range(0, len(item_ids), MAX_LIST_OF_ITEM_IDS_AVAILABILITY_CALL):
        all_available_items.extend(
            await consult_item_available(
                item_ids[i : i + MAX_LIST_OF_ITEM_IDS_AVAILABILITY_CALL]
            )
        )

    # filter items_ids that are not available
    available_items_ids = [
        item_id
        for item_id, available in zip(item_ids, all_available_items)
        if available
    ]
    print(f"Available items: {len(available_items_ids)}")
    if not available_items_ids:
        return []

    # create batches of items according to the MAX_LIST_OF_ITEM_IDS_PRICING_CALL and process them in parallel to get the Prices
    available_item_prices = []
    for i in range(0, len(item_ids), MAX_LIST_OF_ITEM_IDS_PRICING_CALL):
        available_item_prices.extend(
            await consult_price(item_ids[i : i + MAX_LIST_OF_ITEM_IDS_PRICING_CALL])
        )

    # sort items by price
    available_item_prices.sort(key=lambda item: item.selling_price)
    # # return top_k items
    return available_item_prices[:top_k]


if __name__ == "__main__":
    top_k = 10
    # items = ['item_id_1', 'item_id_2', ...]
    items = [f"item_{i}" for i in range(1, 100)]
    # items = []
    print("Total items:", len(items))
    priceditems = asyncio.run(return_top_cheapest_items(items, top_k))
    if not priceditems:
        print("No items available")
        sys.exit(0)
    print()
    print("++++++++++++++++++++++")
    print("Top {} items:".format(top_k))
    print("++++++++++++++++++++++")
    for item in priceditems:
        print(
            f"Item: {item.item_id}, price: {item.selling_price}, discount: {item.discount}"
        )
